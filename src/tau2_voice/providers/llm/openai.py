"""
OpenAI LLM provider with streaming chat completions.

Supports GPT-4o, GPT-4o-mini, and other OpenAI chat models with tool calling.

Install: pip install openai
"""

import json
from typing import AsyncGenerator, Optional, Literal

from loguru import logger
from openai import AsyncOpenAI

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
from tau2_voice.config import VoiceTauConfig


# OpenAI models supporting tool calling
OpenAIModel = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-3.5-turbo",
]


def message_to_openai_format(message: APICompatibleMessage) -> dict:
    """Convert tau2_voice message to OpenAI API format."""
    if isinstance(message, SystemMessage):
        return {
            "role": "system",
            "content": message.content or "",
        }
    elif isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": message.content or "",
        }
    elif isinstance(message, AssistantMessage):
        result = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                }
                for tc in message.tool_calls
            ]
        return result
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.id,
            "content": message.content or "",
        }
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def tools_to_openai_format(tools: list[Tool]) -> list[dict]:
    """Convert tau2_voice tools to OpenAI API format."""
    return [tool.openai_schema for tool in tools]


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI LLM provider with streaming chat completions.
    
    Example:
        provider = OpenAILLMProvider(model="gpt-4o-mini")
        async for chunk in provider.stream_completion(messages, tools):
            print(chunk.content, end="")
    """
    
    def __init__(
        self,
        model: OpenAIModel = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model ID
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate (None for model default)
            api_key: OpenAI API key (defaults to config)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or VoiceTauConfig.OPENAI_API_KEY
        self._client: Optional[AsyncOpenAI] = None
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        self._client = AsyncOpenAI(api_key=self._api_key)
        logger.info(f"OpenAI LLM provider initialized with model: {self.model}")
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("OpenAI LLM provider shut down")
    
    async def stream_completion(
        self,
        messages: list[APICompatibleMessage],
        tools: Optional[list[Tool]] = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Stream a chat completion from OpenAI.
        
        Args:
            messages: Conversation history
            tools: Available tools for function calling
            
        Yields:
            LLMResponse objects with incremental content/tool calls
        """
        if self._client is None:
            await self.initialize()
        
        # Convert messages to OpenAI format
        openai_messages = [message_to_openai_format(m) for m in messages]
        
        # Build request params
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": True,
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        if tools:
            params["tools"] = tools_to_openai_format(tools)
            params["tool_choice"] = "auto"
        
        logger.debug(f"Streaming completion with {len(messages)} messages")
        
        # Track tool calls being built incrementally
        tool_calls_in_progress: dict[int, dict] = {}
        
        try:
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                response = LLMResponse()
                
                # Handle content delta
                if delta.content:
                    response.content = delta.content
                
                # Handle tool calls delta
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        
                        # Initialize or update tool call
                        if idx not in tool_calls_in_progress:
                            tool_calls_in_progress[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        
                        tc = tool_calls_in_progress[idx]
                        
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["arguments"] += tc_delta.function.arguments
                
                # Handle finish reason
                if choice.finish_reason:
                    response.finish_reason = choice.finish_reason
                    
                    # If finished with tool calls, parse them
                    if choice.finish_reason == "tool_calls" and tool_calls_in_progress:
                        for tc_data in tool_calls_in_progress.values():
                            try:
                                arguments = json.loads(tc_data["arguments"])
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tool arguments: {tc_data['arguments']}")
                                arguments = {}
                            
                            response.tool_calls.append(ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                arguments=arguments,
                                requestor="assistant",
                            ))
                
                yield response
                
        except Exception as e:
            logger.error(f"Error during OpenAI streaming: {e}")
            raise
    
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
        if self._client is None:
            await self.initialize()
        
        # Convert messages to OpenAI format
        openai_messages = [message_to_openai_format(m) for m in messages]
        
        # Build request params
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        
        if tools:
            params["tools"] = tools_to_openai_format(tools)
            params["tool_choice"] = "auto"
            params["parallel_tool_calls"] = True
        
        logger.debug(f"Completion with {len(messages)} messages")
        
        try:
            response = await self._client.chat.completions.create(**params)
            choice = response.choices[0]
            
            result = LLMResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason,
            )
            
            # Parse tool calls
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {tc.function.arguments}")
                        arguments = {}
                    
                    result.tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                        requestor="assistant",
                    ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error during OpenAI completion: {e}")
            raise

