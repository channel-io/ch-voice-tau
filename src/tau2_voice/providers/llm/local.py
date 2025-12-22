"""
Local LLM provider using HuggingFace transformers.

Supports local models like nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 with tool calling.

Install: pip install transformers torch accelerate
"""

import json
import asyncio
import re
from typing import AsyncGenerator, Optional, Literal

from loguru import logger

from tau2_voice.providers.base import BaseLLMProvider, LLMResponse
from tau2_voice.models.tool import Tool
from tau2_voice.models.message import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ToolMessage,
    ToolCall,
    APICompatibleMessage,
)


# Default local models
LocalModelID = Literal[
    "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]


def format_tools_for_prompt(tools: list[Tool]) -> str:
    """Format tools into a system prompt description using OpenAI-compatible schema."""
    if not tools:
        return ""
    
    # Use the same OpenAI schema format as Qwen3OmniAgent
    tools_schema = [tool.openai_schema for tool in tools]
    
    return (
        "\n\n# Available Tools\n"
        "You have access to the following tools. When you need to use a tool, respond with a JSON object in this EXACT format:\n"
        '```json\n'
        '{"tool_call": {"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}}\n'
        '```\n\n'
        "You can make MULTIPLE tool calls in a single response if needed. Just include multiple JSON blocks:\n"
        '```json\n'
        '{"tool_call": {"name": "first_function", "arguments": {...}}}\n'
        '{"tool_call": {"name": "second_function", "arguments": {...}}}\n'
        '```\n\n'
        "Tools schema (OpenAI function calling format):\n"
        f"```json\n{json.dumps(tools_schema, indent=2)}\n```\n\n"
        "IMPORTANT: When the user provides information that matches a tool's parameters, you MUST call the appropriate tool. "
        "Do NOT simulate or imagine tool results - actually call the tool and wait for the result."
    )


def message_to_chat_format(message: APICompatibleMessage) -> dict:
    """Convert tau2_voice message to chat format for transformers."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content or ""}
    elif isinstance(message, UserMessage):
        return {"role": "user", "content": message.content or ""}
    elif isinstance(message, AssistantMessage):
        content = message.content or ""
        if message.tool_calls:
            # Format tool calls in content
            for tc in message.tool_calls:
                content += f'\n{{"tool_call": {{"name": "{tc.name}", "arguments": {json.dumps(tc.arguments)}}}}}'
        return {"role": "assistant", "content": content}
    elif isinstance(message, ToolMessage):
        return {
            "role": "user",  # Tool results as user messages
            "content": f"Tool result for {message.id}:\n{message.content or ''}",
        }
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool calls from model output text."""
    tool_calls = []
    
    # Look for JSON tool call patterns
    patterns = [
        r'\{"tool_call":\s*\{[^}]+\}\}',  # {"tool_call": {...}}
        r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}',  # {"name": "...", "arguments": {...}}
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if "tool_call" in data:
                    tc_data = data["tool_call"]
                else:
                    tc_data = data
                
                if "name" in tc_data:
                    tool_calls.append(ToolCall(
                        id=f"local_{hash(match) % 10000}",
                        name=tc_data["name"],
                        arguments=tc_data.get("arguments", {}),
                        requestor="assistant",
                    ))
            except json.JSONDecodeError:
                continue
    
    return tool_calls


class LocalLLMProvider(BaseLLMProvider):
    """
    Local LLM provider using HuggingFace transformers pipeline.
    
    Example:
        provider = LocalLLMProvider(model_id="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1")
        await provider.initialize()
        async for chunk in provider.stream_completion(messages, tools):
            print(chunk.content, end="")
    """
    
    def __init__(
        self,
        model_id: str = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
        thinking: Literal["on", "off"] = "off",
    ):
        """
        Initialize local LLM provider.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to run on ("cpu", "cuda", or "auto")
            torch_dtype: Data type for model weights
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling (False for greedy decoding)
            thinking: Enable/disable reasoning mode for Nemotron models ("on" or "off")
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.thinking = thinking
        
        self._pipe = None
        self._tokenizer = None
        self._device = None
    
    async def initialize(self) -> None:
        """Load the local LLM model using pipeline."""
        try:
            import torch
            import transformers
        except ImportError:
            raise ImportError(
                "transformers or torch not installed. Run: pip install transformers torch accelerate"
            )
        
        logger.info(f"Loading local LLM: {self.model_id} on {self.device}")
        
        # Determine dtype
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Load model in thread pool
        loop = asyncio.get_event_loop()
        
        def load_model():
            # Use official HuggingFace pipeline approach
            model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
            
            # Use slow tokenizer to avoid tokenizers library compatibility issues
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # When thinking is off, use do_sample=False as per official Nemotron docs
            effective_do_sample = self.do_sample if self.thinking == "on" else False
            
            pipe_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                **model_kwargs
            }
            
            # Only add sampling parameters when do_sample=True
            if effective_do_sample:
                pipe_kwargs["temperature"] = self.temperature
                pipe_kwargs["top_p"] = self.top_p
                pipe_kwargs["do_sample"] = True
            else:
                pipe_kwargs["do_sample"] = False
            
            pipe = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                tokenizer=tokenizer,
                **pipe_kwargs
            )
            
            return pipe, tokenizer
        
        self._pipe, self._tokenizer = await loop.run_in_executor(None, load_model)
        
        logger.info(f"Local LLM loaded: {self.model_id}")
    
    async def shutdown(self) -> None:
        """Release model resources."""
        self._pipe = None
        self._tokenizer = None
        logger.info("Local LLM unloaded")
    
    async def stream_completion(
        self,
        messages: list[APICompatibleMessage],
        tools: Optional[list[Tool]] = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Stream a chat completion from local LLM.
        
        Note: Pipeline doesn't support true streaming, so we yield the full response.
        
        Args:
            messages: Conversation history
            tools: Available tools for function calling
            
        Yields:
            LLMResponse objects with content/tool calls
        """
        if self._pipe is None:
            await self.initialize()
        
        # Convert messages to chat format
        chat_messages = [message_to_chat_format(m) for m in messages]
        
        # Add tool descriptions to system message
        if tools:
            tool_prompt = format_tools_for_prompt(tools)
            if chat_messages and chat_messages[0]["role"] == "system":
                chat_messages[0]["content"] += tool_prompt
            else:
                chat_messages.insert(0, {"role": "system", "content": tool_prompt})
        
        # Handle Nemotron thinking mode
        thinking_prefix = f"detailed thinking {self.thinking}\n\n"
        if chat_messages and chat_messages[0]["role"] == "system":
            chat_messages[0]["content"] = thinking_prefix + chat_messages[0]["content"]
        else:
            chat_messages.insert(0, {"role": "system", "content": thinking_prefix.strip()})
        
        # Pre-fill assistant response to prevent emergent thinking when off
        if self.thinking == "off":
            chat_messages.append({"role": "assistant", "content": "<think>\n</think>"})
        
        logger.debug(f"Generating with {len(messages)} messages (thinking={self.thinking})")
        
        # Log the system prompt for debugging
        if chat_messages and chat_messages[0]["role"] == "system":
            system_content = chat_messages[0]["content"]
            logger.info(f"[LLM SYSTEM PROMPT] (first 500 chars):\n{system_content[:500]}...")
        
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        
        def generate():
            result = self._pipe(chat_messages)
            # Pipeline returns list of dicts with 'generated_text'
            if result and len(result) > 0:
                generated = result[0].get("generated_text", [])
                # Get the assistant's response (last message)
                if isinstance(generated, list) and len(generated) > 0:
                    last_msg = generated[-1]
                    if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
                        return last_msg.get("content", "")
                elif isinstance(generated, str):
                    return generated
            return ""
        
        text = await loop.run_in_executor(None, generate)
        
        # Log raw LLM output
        logger.info(f"[LLM RAW OUTPUT]\n{text}")
        
        # Remove <think>...</think> tags from output (Nemotron thinking mode)
        import re
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        
        # Yield the full text
        if text:
            yield LLMResponse(content=text)
        
        # Parse tool calls from output
        tool_calls = parse_tool_calls(text) if tools else []
        
        if tool_calls:
            yield LLMResponse(
                tool_calls=tool_calls,
                finish_reason="tool_calls",
            )
        else:
            yield LLMResponse(finish_reason="stop")
    
    async def completion(
        self,
        messages: list[APICompatibleMessage],
        tools: Optional[list[Tool]] = None,
    ) -> LLMResponse:
        """
        Non-streaming chat completion.
        
        Args:
            messages: Conversation history
            tools: Available tools for function calling
            
        Returns:
            Complete LLMResponse
        """
        if self._pipe is None:
            await self.initialize()
        
        # Convert messages to chat format
        chat_messages = [message_to_chat_format(m) for m in messages]
        
        # Add tool descriptions to system message
        if tools:
            tool_prompt = format_tools_for_prompt(tools)
            if chat_messages and chat_messages[0]["role"] == "system":
                chat_messages[0]["content"] += tool_prompt
            else:
                chat_messages.insert(0, {"role": "system", "content": tool_prompt})
        
        # Handle Nemotron thinking mode
        thinking_prefix = f"detailed thinking {self.thinking}\n\n"
        if chat_messages and chat_messages[0]["role"] == "system":
            chat_messages[0]["content"] = thinking_prefix + chat_messages[0]["content"]
        else:
            chat_messages.insert(0, {"role": "system", "content": thinking_prefix.strip()})
        
        # Pre-fill assistant response to prevent emergent thinking when off
        if self.thinking == "off":
            chat_messages.append({"role": "assistant", "content": "<think>\n</think>"})
        
        logger.debug(f"Generating with {len(messages)} messages (thinking={self.thinking})")
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        
        def generate():
            result = self._pipe(chat_messages)
            # Pipeline returns list of dicts with 'generated_text'
            if result and len(result) > 0:
                generated = result[0].get("generated_text", [])
                # Get the assistant's response (last message)
                if isinstance(generated, list) and len(generated) > 0:
                    last_msg = generated[-1]
                    if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
                        return last_msg.get("content", "")
                elif isinstance(generated, str):
                    return generated
            return ""
        
        text = await loop.run_in_executor(None, generate)
        
        # Remove <think>...</think> tags from output (Nemotron thinking mode)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        
        # Parse tool calls
        tool_calls = parse_tool_calls(text) if tools else []
        
        # Remove tool call JSON from content
        content = text
        for tc in tool_calls:
            # Remove the JSON from content
            content = re.sub(r'\{"tool_call":\s*\{[^}]+\}\}', '', content)
            content = re.sub(r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}', '', content)
        
        return LLMResponse(
            content=content.strip(),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

