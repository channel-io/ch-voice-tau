"""
Base protocol classes for cascade pipeline providers.

Each provider type (ASR, LLM, TTS) has an abstract base class that defines
the interface. Implementations can be swapped easily for different backends
(local, SaaS APIs, etc.).

Uses existing message models from tau2_voice.models.message for compatibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Any

from tau2_voice.models.tool import Tool
from tau2_voice.models.message import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ToolMessage,
    ToolCall,
    APICompatibleMessage,
)


@dataclass
class LLMResponse:
    """
    Response from LLM generation (streaming chunk or final response).
    Uses existing ToolCall model for compatibility.
    """
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None


class BaseASRProvider(ABC):
    """
    Base class for ASR (Automatic Speech Recognition) providers.
    
    Implementations should convert audio data to text transcription.
    """
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int = 24000) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw PCM audio data (16-bit signed, mono)
            sample_rate: Sample rate of the audio in Hz
            
        Returns:
            Transcribed text
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize the provider (load models, etc.). Override if needed."""
        pass
    
    async def shutdown(self) -> None:
        """Cleanup resources. Override if needed."""
        pass


class BaseLLMProvider(ABC):
    """
    Base class for LLM (Large Language Model) providers.
    
    Implementations should support streaming chat completions with tool calling.
    Uses existing message models from tau2_voice.models.message.
    """
    
    @abstractmethod
    async def stream_completion(
        self,
        messages: list[APICompatibleMessage],
        tools: Optional[list[Tool]] = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Stream a chat completion.
        
        Args:
            messages: Conversation history using existing message models
            tools: Available tools for function calling
            
        Yields:
            LLMResponse objects with incremental content/tool calls
        """
        pass
    
    async def completion(
        self,
        messages: list[APICompatibleMessage],
        tools: Optional[list[Tool]] = None,
    ) -> LLMResponse:
        """
        Non-streaming chat completion. Default implementation collects stream.
        
        Args:
            messages: Conversation history
            tools: Available tools for function calling
            
        Returns:
            Complete LLMResponse
        """
        full_response = LLMResponse()
        async for chunk in self.stream_completion(messages, tools):
            full_response.content += chunk.content
            if chunk.tool_calls:
                full_response.tool_calls.extend(chunk.tool_calls)
            if chunk.finish_reason:
                full_response.finish_reason = chunk.finish_reason
        return full_response
    
    async def initialize(self) -> None:
        """Initialize the provider. Override if needed."""
        pass
    
    async def shutdown(self) -> None:
        """Cleanup resources. Override if needed."""
        pass


class BaseTTSProvider(ABC):
    """
    Base class for TTS (Text-to-Speech) providers.
    
    Implementations should convert text to audio, preferably with streaming support.
    """
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis for the given text.
        
        Args:
            text: Text to synthesize
            
        Yields:
            PCM audio chunks (16-bit signed, mono)
        """
        pass
    
    async def synthesize(self, text: str) -> bytes:
        """
        Non-streaming synthesis. Default implementation collects stream.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Complete PCM audio data
        """
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the provider. Override if needed."""
        pass
    
    async def shutdown(self) -> None:
        """Cleanup resources. Override if needed."""
        pass
