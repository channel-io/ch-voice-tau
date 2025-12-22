"""
OpenAI TTS provider with streaming audio synthesis.

Supports gpt-4o-mini-tts and tts-1/tts-1-hd models.

Install: pip install openai
"""

import re
from typing import AsyncGenerator, Optional, Literal

from loguru import logger
from openai import AsyncOpenAI

from tau2_voice.providers.base import BaseTTSProvider
from tau2_voice.config import VoiceTauConfig


# Available TTS models
TTSModel = Literal["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]

# Available voices
TTSVoice = Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]


def clean_text_for_tts(text: str) -> str:
    """
    Clean markdown and special characters for better TTS output.
    
    Reused from qwen3_omni.py for consistency.
    """
    # **bold** -> bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # *italic* -> italic
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # - list items -> remove dash
    text = re.sub(r'^[\s]*[-•]\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown headers (##, ###, etc)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    
    return text.strip()


class OpenAITTSProvider(BaseTTSProvider):
    """
    OpenAI TTS provider with streaming audio synthesis.
    
    Example:
        provider = OpenAITTSProvider(model="gpt-4o-mini-tts", voice="alloy")
        async for audio_chunk in provider.synthesize_stream("Hello world!"):
            # Process PCM audio chunk
            pass
    """
    
    def __init__(
        self,
        model: TTSModel = "gpt-4o-mini-tts",
        voice: TTSVoice = "alloy",
        instructions: Optional[str] = "Speak clearly and naturally.",
        speed: float = 1.0,
        api_key: Optional[str] = None,
        output_sample_rate: int = 24000,
    ):
        """
        Initialize OpenAI TTS provider.
        
        Args:
            model: TTS model to use
            voice: Voice to use for synthesis
            instructions: Voice instructions (for gpt-4o-mini-tts)
            speed: Speech speed (0.25 to 4.0)
            api_key: OpenAI API key (defaults to config)
            output_sample_rate: Output sample rate (24000 for OpenAI TTS)
        """
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.speed = speed
        self._api_key = api_key or VoiceTauConfig.OPENAI_API_KEY
        self._sample_rate = output_sample_rate
        self._client: Optional[AsyncOpenAI] = None
    
    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._sample_rate
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        self._client = AsyncOpenAI(api_key=self._api_key)
        logger.info(f"OpenAI TTS provider initialized with model: {self.model}, voice: {self.voice}")
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("OpenAI TTS provider shut down")
    
    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis for the given text.
        
        Args:
            text: Text to synthesize
            
        Yields:
            PCM audio chunks (16-bit signed, mono, 24kHz)
        """
        if self._client is None:
            await self.initialize()
        
        if not text or not text.strip():
            return
        
        # Clean text for better TTS output
        cleaned_text = clean_text_for_tts(text)
        
        if not cleaned_text:
            return
        
        logger.debug(f"Synthesizing TTS: {cleaned_text[:50]}...")
        
        try:
            # Build request params
            params = {
                "model": self.model,
                "voice": self.voice,
                "input": cleaned_text,
                "response_format": "pcm",  # Raw PCM audio
            }
            
            # Add instructions for gpt-4o-mini-tts
            if self.model == "gpt-4o-mini-tts" and self.instructions:
                params["instructions"] = self.instructions
            
            # Add speed if not default
            if self.speed != 1.0:
                params["speed"] = self.speed
            
            # Stream response
            async with self._client.audio.speech.with_streaming_response.create(**params) as response:
                chunk_size = 4096  # bytes
                
                async for chunk in response.iter_bytes(chunk_size):
                    yield chunk
                    
            logger.debug(f"TTS synthesis completed for: {cleaned_text[:30]}...")
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise
    
    async def synthesize(self, text: str) -> bytes:
        """
        Non-streaming synthesis.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Complete PCM audio data
        """
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)

