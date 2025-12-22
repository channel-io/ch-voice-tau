"""
Cascade Agent: ASR -> LLM -> TTS pipeline for speech-to-speech interaction.

This agent composes pluggable ASR, LLM, and TTS providers to create a
turn-based speech-to-speech system compatible with the BaseAgent interface.

The streaming pipeline:
1. Collect audio chunks until turn end
2. ASR: Transcribe audio to text
3. LLM: Generate response (streaming)
4. TTS: Convert response to audio (streaming per sentence)
"""

import asyncio
import base64
import re
from typing import Optional, Literal, AsyncGenerator
from typing_extensions import override

from loguru import logger

from tau2_voice.agent.base import BaseAgent
from tau2_voice.models.tool import Tool
from tau2_voice.models.events import (
    Event,
    AudioChunkEvent,
    AudioDoneEvent,
    TranscriptUpdateEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
    SpeakRequestEvent,
)
from tau2_voice.models.message import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ToolMessage,
    ToolCall,
)
from tau2_voice.providers.base import BaseASRProvider, BaseLLMProvider, BaseTTSProvider


# Sentence boundary pattern for streaming TTS
SENTENCE_END_PATTERN = re.compile(r'[.!?]\s*$|[.!?]\s+')


def is_sentence_boundary(text: str, min_length: int = 30) -> bool:
    """Check if text ends at a sentence boundary."""
    if len(text) < min_length:
        return False
    return bool(SENTENCE_END_PATTERN.search(text))


class CascadeAgent(BaseAgent):
    """
    Cascade pipeline agent: ASR -> LLM -> TTS.
    
    This agent is compatible with the existing BaseAgent interface and can be
    used as a drop-in replacement for RealtimeAgent, GeminiLiveAgent, etc.
    
    Example:
        from tau2_voice.providers.asr import WhisperLocalProvider
        from tau2_voice.providers.llm import OpenAILLMProvider
        from tau2_voice.providers.tts import OpenAITTSProvider
        
        agent = CascadeAgent(
            tools=tools,
            domain_policy=policy,
            asr_provider=WhisperLocalProvider(),
            llm_provider=OpenAILLMProvider(model="gpt-4o-mini"),
            tts_provider=OpenAITTSProvider(voice="alloy"),
        )
    """
    
    def __init__(
        self,
        tools: Optional[list[Tool]],
        domain_policy: Optional[str],
        asr_provider: BaseASRProvider,
        llm_provider: BaseLLMProvider,
        tts_provider: BaseTTSProvider,
        role: Literal["user", "assistant"] = "assistant",
        input_sample_rate: int = 24000,
        stream_tts_by_sentence: bool = True,
        min_sentence_length: int = 30,
    ):
        """
        Initialize cascade agent.
        
        Args:
            tools: Available tools for the agent
            domain_policy: System prompt / policy for the LLM
            asr_provider: ASR provider for speech-to-text
            llm_provider: LLM provider for text generation
            tts_provider: TTS provider for text-to-speech
            role: Agent role ("user" or "assistant")
            input_sample_rate: Expected input audio sample rate
            stream_tts_by_sentence: Stream TTS by sentence for lower latency
            min_sentence_length: Minimum chars before sentence boundary triggers TTS
        """
        super().__init__(tools=tools, domain_policy=domain_policy, role=role)
        
        self.asr_provider = asr_provider
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider
        self.input_sample_rate = input_sample_rate
        self.stream_tts_by_sentence = stream_tts_by_sentence
        self.min_sentence_length = min_sentence_length
        
        # Audio buffer for incoming chunks
        self._audio_buffer: list[bytes] = []
        
        # Message history for LLM
        self._messages: list = []
        
        # Event queue for async generator
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._is_connected = False
        
        # Track pending tool calls (by ID)
        self._pending_tool_call_ids: set[str] = set()
        self._message_counter = 0
        
        # Lock to serialize message list access and response generation
        self._response_lock = asyncio.Lock()
    
    @property
    def system_prompt(self) -> str:
        """Get system prompt from domain policy."""
        return self.domain_policy or "You are a helpful assistant."
    
    @override
    async def connect(self):
        """Initialize all providers."""
        logger.info(f"[{self.role}] Connecting cascade agent...")
        
        # Initialize providers
        await self.asr_provider.initialize()
        await self.llm_provider.initialize()
        await self.tts_provider.initialize()
        
        self._is_connected = True
        
        # Initialize message history with system prompt
        self._messages = [
            SystemMessage(role="system", content=self.system_prompt)
        ]
        
        logger.info(f"[{self.role}] Cascade agent connected")
    
    @override
    async def disconnect(self):
        """Shutdown all providers."""
        self._is_connected = False
        
        await self.asr_provider.shutdown()
        await self.llm_provider.shutdown()
        await self.tts_provider.shutdown()
        
        logger.info(f"[{self.role}] Cascade agent disconnected")
    
    @override
    async def publish(self, event: Event):
        """
        Handle incoming events.
        
        - audio.chunk: Buffer audio data
        - audio.done: Process turn (ASR -> LLM -> TTS)
        - speak.request: Generate response without audio input
        - tool_call.result: Add result to messages and continue
        """
        if not self._is_connected:
            return
        
        if event.type == "audio.chunk":
            # Decode and buffer audio
            audio_bytes = base64.b64decode(event.audio_chunk)
            self._audio_buffer.append(audio_bytes)
            
        elif event.type == "audio.done":
            # Process the turn
            await self._process_turn()
            
        elif event.type == "speak.request":
            # Generate response without audio (e.g., after tool calls)
            # Only process if all pending tool calls have been responded to
            if len(self._pending_tool_call_ids) == 0 and hasattr(event, 'instructions') and event.instructions:
                # Direct instruction (e.g., kickoff prompt)
                await self._process_instruction(event.instructions)
            # Note: Don't auto-trigger after tool calls here; wait for all responses
                
        elif event.type == "tool_call.result":
            # Acquire lock for message list modification
            async with self._response_lock:
                # Add tool result to messages
                tool_message = ToolMessage(
                    id=event.id,
                    role="tool",
                    content=event.content,
                    requestor="assistant",
                )
                self._messages.append(tool_message)
                
                # Remove from pending set
                if event.id in self._pending_tool_call_ids:
                    self._pending_tool_call_ids.discard(event.id)
                    logger.debug(f"Received tool result for {event.id}, remaining: {len(self._pending_tool_call_ids)}")
                
                # If all pending tool calls are complete, generate response
                if len(self._pending_tool_call_ids) == 0:
                    logger.debug("All tool calls complete, generating response")
                    await self._process_turn_after_tool()
            
        elif event.type == "transcript.update":
            # CascadeAgent uses audio input (ASR) instead of text transcripts.
            # Ignore transcript.update events to avoid duplicate responses.
            # Text-based agents like Gemini can use this for text-only interaction.
            logger.debug(f"[{self.role}] Ignoring transcript.update event (using audio/ASR instead)")
    
    @override
    async def subscribe(self) -> AsyncGenerator[Event, None]:
        """Subscribe to outbound events from this agent."""
        while self._is_connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue
    
    async def _process_turn(self):
        """Process a complete turn: ASR -> LLM -> TTS."""
        if not self._audio_buffer:
            logger.warning("No audio data to process")
            return
        
        # Combine audio chunks first (before lock)
        audio_data = b"".join(self._audio_buffer)
        self._audio_buffer = []
        
        logger.debug(f"Processing {len(audio_data)} bytes of audio")
        
        # Step 1: ASR - Transcribe audio (can happen outside lock)
        try:
            transcript = await self.asr_provider.transcribe(
                audio_data, 
                sample_rate=self.input_sample_rate
            )
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return
        
        if not transcript.strip():
            logger.warning("Empty transcript from ASR")
            # Send empty audio done to continue conversation
            await self._event_queue.put(AudioDoneEvent(
                role=self.role,
                message_id="empty_transcript"
            ))
            return
        
        logger.info(f"[{self.role}] ASR transcript: {transcript}")
        
        # Acquire lock for message list modification and LLM call
        async with self._response_lock:
            # If tool calls are pending, don't process new turns yet
            # This prevents message sequence errors with OpenAI API
            if len(self._pending_tool_call_ids) > 0:
                logger.warning(f"Skipping turn processing: {len(self._pending_tool_call_ids)} tool calls pending")
                return
            
            # Add user message to history
            user_message = UserMessage(role="user", content=transcript)
            self._messages.append(user_message)
            
            # Step 2 & 3: LLM -> TTS (streaming)
            await self._generate_and_stream_response()
    
    async def _process_turn_after_tool(self):
        """Process turn after receiving tool results."""
        logger.debug("Processing turn after tool call")
        # Lock is already held when this is called from publish()
        # But we call _generate_and_stream_response which expects lock to be held
        await self._generate_and_stream_response()
    
    async def _process_instruction(self, instruction: str):
        """Process a direct instruction (e.g., kickoff prompt)."""
        logger.debug(f"Processing instruction: {instruction[:50]}...")
        
        # Acquire lock for message list modification and LLM call
        async with self._response_lock:
            # Add instruction as user message
            user_message = UserMessage(role="user", content=instruction)
            self._messages.append(user_message)
            
            await self._generate_and_stream_response()
    
    async def _generate_and_stream_response(self):
        """Generate LLM response and stream to TTS."""
        self._message_counter += 1
        message_id = f"cascade_{self._message_counter}"
        
        # Track full response for message history
        full_content = ""
        tool_calls: list[ToolCall] = []
        
        # Buffer for sentence-based TTS streaming
        tts_buffer = ""
        
        try:
            # Stream LLM response
            async for chunk in self.llm_provider.stream_completion(
                self._messages,
                tools=self.tools,
            ):
                # Handle content
                if chunk.content:
                    full_content += chunk.content
                    tts_buffer += chunk.content
                    
                    # Stream TTS by sentence for lower latency
                    if self.stream_tts_by_sentence:
                        if is_sentence_boundary(tts_buffer, self.min_sentence_length):
                            await self._stream_tts_chunk(tts_buffer, message_id)
                            tts_buffer = ""
                
                # Handle tool calls
                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
            
            # Stream remaining TTS buffer
            if tts_buffer.strip():
                await self._stream_tts_chunk(tts_buffer, message_id)
            
            # Send transcript update
            if full_content:
                await self._event_queue.put(TranscriptUpdateEvent(
                    role=self.role,
                    message_id=message_id,
                    transcript=full_content,
                ))
            
            # Handle tool calls
            if tool_calls:
                # Add assistant message with tool calls to history
                assistant_message = AssistantMessage(
                    role="assistant",
                    content=full_content,
                    tool_calls=tool_calls,
                )
                self._messages.append(assistant_message)
                
                # Track pending tool calls
                for tc in tool_calls:
                    self._pending_tool_call_ids.add(tc.id)
                logger.debug(f"Added {len(tool_calls)} tool calls to pending: {[tc.id for tc in tool_calls]}")
                
                # Emit tool call events
                for tc in tool_calls:
                    await self._event_queue.put(ToolCallRequestEvent(
                        role=self.role,
                        message_id=message_id,
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.arguments,
                        requestor="assistant",
                    ))
            else:
                # Add assistant message to history (no tool calls)
                assistant_message = AssistantMessage(
                    role="assistant",
                    content=full_content,
                )
                self._messages.append(assistant_message)
            
            # Send audio done
            await self._event_queue.put(AudioDoneEvent(
                role=self.role,
                message_id=message_id,
            ))
            
            logger.debug(f"Response completed: {full_content[:100]}...")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            
            # Send audio done even on error
            await self._event_queue.put(AudioDoneEvent(
                role=self.role,
                message_id=message_id,
            ))
    
    async def _stream_tts_chunk(self, text: str, message_id: str):
        """Stream TTS for a text chunk."""
        if not text.strip():
            return
        
        try:
            async for audio_chunk in self.tts_provider.synthesize_stream(text):
                # Encode as base64
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                await self._event_queue.put(AudioChunkEvent(
                    role=self.role,
                    message_id=message_id,
                    audio_chunk=audio_b64,
                ))
        except Exception as e:
            logger.error(f"TTS error: {e}")

