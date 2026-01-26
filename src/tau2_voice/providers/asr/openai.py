"""
OpenAI Whisper ASR provider using the Audio Transcriptions API.

Uses the OpenAI Whisper API for cloud-based speech recognition.

Install: pip install openai
"""

import asyncio
import io
from typing import Optional, Literal

from loguru import logger
from openai import AsyncOpenAI

from tau2_voice.providers.base import BaseASRProvider
from tau2_voice.config import VoiceTauConfig


# Available Whisper models via OpenAI API
WhisperAPIModel = Literal["whisper-1"]


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Convert raw PCM data to WAV format bytes.
    
    Args:
        pcm_data: Raw PCM bytes (16-bit signed, mono)
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        bits_per_sample: Bits per sample (16 for 16-bit audio)
        
    Returns:
        WAV format bytes with header
    """
    import struct
    
    # WAV header parameters
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    file_size = 36 + data_size  # 36 bytes header + data
    
    # Build WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        file_size,         # ChunkSize
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (16 for PCM)
        1,                 # AudioFormat (1 for PCM)
        channels,          # NumChannels
        sample_rate,       # SampleRate
        byte_rate,         # ByteRate
        block_align,       # BlockAlign
        bits_per_sample,   # BitsPerSample
        b'data',           # Subchunk2ID
        data_size,         # Subchunk2Size
    )
    
    return header + pcm_data


class OpenAIASRProvider(BaseASRProvider):
    """
    OpenAI Whisper ASR provider using the Audio Transcriptions API.
    
    Example:
        provider = OpenAIASRProvider(language="en")
        await provider.initialize()
        transcript = await provider.transcribe(audio_data)
    """
    
    def __init__(
        self,
        model: WhisperAPIModel = "whisper-1",
        language: Optional[str] = "en",
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Initialize OpenAI ASR provider.
        
        Args:
            model: Whisper model to use (currently only "whisper-1")
            language: Language code (e.g., "en") or None for auto-detection
            api_key: OpenAI API key (defaults to config)
            prompt: Optional prompt to guide transcription style/vocabulary
        """
        self.model = model
        self.language = language
        self._api_key = api_key or VoiceTauConfig.OPENAI_API_KEY
        self.prompt = prompt
        self._client: Optional[AsyncOpenAI] = None
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        self._client = AsyncOpenAI(api_key=self._api_key)
        logger.info(f"OpenAI ASR provider initialized with model: {self.model}, language: {self.language}")
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("OpenAI ASR provider shut down")
    
    async def transcribe(self, audio_data: bytes, sample_rate: int = 24000) -> str:
        """
        Transcribe audio data to text using OpenAI Whisper API.
        
        Args:
            audio_data: Raw PCM audio data (16-bit signed, mono)
            sample_rate: Sample rate of the audio in Hz
            
        Returns:
            Transcribed text
        """
        if self._client is None:
            await self.initialize()
        
        if not audio_data:
            return ""
        
        # Convert PCM to WAV (OpenAI API requires file format)
        wav_data = pcm_to_wav_bytes(audio_data, sample_rate=sample_rate)
        
        # Create file-like object for API
        audio_file = io.BytesIO(wav_data)
        audio_file.name = "audio.wav"
        
        try:
            # Build request params
            params = {
                "model": self.model,
                "file": audio_file,
                "response_format": "text",
            }
            
            if self.language:
                params["language"] = self.language
            
            if self.prompt:
                params["prompt"] = self.prompt
            
            # Call API
            transcript = await self._client.audio.transcriptions.create(**params)
            
            logger.debug(f"Transcribed: {transcript[:100]}...")
            
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            raise
