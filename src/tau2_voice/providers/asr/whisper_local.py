"""
Local Whisper ASR provider using HuggingFace transformers pipeline.

Uses the official openai/whisper-large-v3 model from HuggingFace.

Install: pip install transformers torch
"""

import asyncio
import struct
from typing import Literal, Optional

from loguru import logger

from tau2_voice.providers.base import BaseASRProvider

# Model IDs available on HuggingFace
WhisperModelID = Literal[
    "openai/whisper-tiny",
    "openai/whisper-tiny.en",
    "openai/whisper-base",
    "openai/whisper-base.en",
    "openai/whisper-small",
    "openai/whisper-small.en",
    "openai/whisper-medium",
    "openai/whisper-medium.en",
    "openai/whisper-large",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
]


def pcm_to_float32(pcm_data: bytes, sample_rate: int = 24000):
    """
    Convert raw PCM 16-bit signed audio to float32 numpy array.
    
    Args:
        pcm_data: Raw PCM bytes (16-bit signed, mono)
        sample_rate: Sample rate of the audio
        
    Returns:
        Dict with 'array' and 'sampling_rate' keys (compatible with pipeline)
    """
    import numpy as np
    
    # Unpack 16-bit signed integers
    num_samples = len(pcm_data) // 2
    samples = struct.unpack(f'<{num_samples}h', pcm_data)
    
    # Convert to float32 in range [-1.0, 1.0]
    audio_array = np.array(samples, dtype=np.float32) / 32768.0
    
    return {"array": audio_array, "sampling_rate": sample_rate}


class WhisperLocalProvider(BaseASRProvider):
    """
    Local Whisper ASR provider using HuggingFace transformers pipeline.
    
    Example:
        provider = WhisperLocalProvider(model_id="openai/whisper-large-v3")
        await provider.initialize()
        transcript = await provider.transcribe(audio_data)
    """
    
    def __init__(
        self,
        model_id: WhisperModelID = "openai/whisper-large-v3",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        torch_dtype: Literal["float16", "float32", "auto"] = "auto",
        language: Optional[str] = "en",
    ):
        """
        Initialize Whisper provider.
        
        Args:
            model_id: HuggingFace model ID (e.g., "openai/whisper-large-v3")
            device: Device to run on ("cpu", "cuda", or "auto")
            torch_dtype: Data type for model weights
            language: Language code (e.g., "en") or None for auto-detection
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.language = language
        
        self._pipe = None
        self._device = None
        self._dtype = None
    
    async def initialize(self) -> None:
        """Load the Whisper model from HuggingFace using pipeline."""
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as e:
            raise ImportError(
                "transformers or torch not installed. Run: pip install transformers torch"
            )
        
        logger.info(f"Loading Whisper model: {self.model_id} on {self.device}")
        
        # Determine device
        if self.device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device
        
        # Determine dtype
        if self.torch_dtype == "auto":
            self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.torch_dtype == "float16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32
        
        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def load_model():
            # Load model using official HuggingFace approach
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self._device)
            
            processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Create pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self._dtype,
                device=self._device,
            )
            
            return pipe
        
        self._pipe = await loop.run_in_executor(None, load_model)
        
        logger.info(f"Whisper model loaded: {self.model_id} on {self._device}")
    
    async def shutdown(self) -> None:
        """Release model resources."""
        self._pipe = None
        logger.info("Whisper model unloaded")
    
    async def transcribe(self, audio_data: bytes, sample_rate: int = 24000) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw PCM audio data (16-bit signed, mono)
            sample_rate: Sample rate of the audio in Hz
            
        Returns:
            Transcribed text
        """
        if self._pipe is None:
            await self.initialize()
        
        if not audio_data:
            return ""
        
        # Convert PCM to format expected by pipeline
        audio_input = pcm_to_float32(audio_data, sample_rate)
        
        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        
        def run_transcription():
            # Build generation kwargs
            generate_kwargs = {}
            
            # Set language if specified
            if self.language:
                generate_kwargs["language"] = self.language
                generate_kwargs["task"] = "transcribe"
            
            result = self._pipe(
                audio_input,
                generate_kwargs=generate_kwargs if generate_kwargs else None,
            )
            
            return result["text"]
        
        transcript = await loop.run_in_executor(None, run_transcription)
        
        logger.debug(f"Transcribed: {transcript[:100]}...")
        
        return transcript.strip()
