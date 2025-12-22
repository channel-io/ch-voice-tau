"""
Chatterbox TTS provider - Local text-to-speech using Resemble AI's Chatterbox model.

Chatterbox is a production-grade open source TTS model with:
- SoTA zero-shot English TTS
- Multilingual support (23 languages)
- Emotion exaggeration control
- 0.5B Llama backbone

Install: pip install chatterbox-tts
"""

import asyncio
import struct
from typing import AsyncGenerator, Optional, Literal

from loguru import logger

from tau2_voice.providers.base import BaseTTSProvider


# Supported languages for multilingual model
ChatterboxLanguage = Literal[
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh"
]


def wav_to_pcm_24k(wav_tensor, source_sr: int) -> bytes:
    """
    Convert wav tensor to PCM bytes at 24kHz.
    
    Args:
        wav_tensor: Torch tensor of audio samples
        source_sr: Source sample rate
        
    Returns:
        PCM bytes (16-bit signed, mono, 24kHz)
    """
    import torch
    import torchaudio.functional as F
    
    # Ensure mono
    if wav_tensor.dim() > 1 and wav_tensor.shape[0] > 1:
        wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
    elif wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    
    # Resample to 24kHz if needed
    if source_sr != 24000:
        wav_tensor = F.resample(wav_tensor, source_sr, 24000)
    
    # Convert to 16-bit PCM
    wav_tensor = wav_tensor.squeeze()
    wav_tensor = torch.clamp(wav_tensor, -1.0, 1.0)
    pcm_samples = (wav_tensor * 32767).to(torch.int16)
    
    # Convert to bytes
    return pcm_samples.cpu().numpy().tobytes()


class ChatterboxTTSProvider(BaseTTSProvider):
    """
    Local TTS provider using Resemble AI's Chatterbox model.
    
    Example:
        provider = ChatterboxTTSProvider()
        await provider.initialize()
        async for audio_chunk in provider.synthesize_stream("Hello world!"):
            # Process PCM audio chunk
            pass
    """
    
    def __init__(
        self,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        audio_prompt_path: Optional[str] = None,
        chunk_size: int = 4096,
    ):
        """
        Initialize Chatterbox TTS provider.
        
        Args:
            device: Device to run on ("cpu", "cuda", or "auto")
            exaggeration: Emotion exaggeration control (0.0 to 1.0+)
                         Higher values = more expressive, faster speech
            cfg_weight: Classifier-free guidance weight (0.0 to 1.0)
                       Lower values (~0.3) = slower, more deliberate pacing
            audio_prompt_path: Path to reference audio for voice cloning (optional)
            chunk_size: Size of audio chunks to yield (in bytes)
        """
        self.device = device
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.audio_prompt_path = audio_prompt_path
        self.chunk_size = chunk_size
        
        self._model = None
        self._device = None
        self._sample_rate = 24000  # Output sample rate
    
    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._sample_rate
    
    async def initialize(self) -> None:
        """Load the Chatterbox model."""
        try:
            import torch
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise ImportError(
                "chatterbox-tts not installed. Run: pip install chatterbox-tts"
            )
        
        # Determine device
        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device
        
        logger.info(f"Loading Chatterbox TTS model on {self._device}...")
        
        # Load model in thread pool
        loop = asyncio.get_event_loop()
        
        def load_model():
            return ChatterboxTTS.from_pretrained(device=self._device)
        
        self._model = await loop.run_in_executor(None, load_model)
        
        # Store the model's native sample rate
        self._model_sr = self._model.sr
        
        logger.info(f"Chatterbox TTS loaded on {self._device} (native sr={self._model_sr})")
    
    async def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("Chatterbox TTS unloaded")
    
    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis for the given text.
        
        Note: Chatterbox generates complete audio, so we chunk the output
        for streaming compatibility.
        
        Args:
            text: Text to synthesize
            
        Yields:
            PCM audio chunks (16-bit signed, mono, 24kHz)
        """
        if self._model is None:
            await self.initialize()
        
        if not text or not text.strip():
            return
        
        logger.debug(f"Synthesizing with Chatterbox: {text[:50]}...")
        
        # Generate audio in thread pool
        loop = asyncio.get_event_loop()
        
        def generate():
            kwargs = {
                "exaggeration": self.exaggeration,
                "cfg_weight": self.cfg_weight,
            }
            
            if self.audio_prompt_path:
                kwargs["audio_prompt_path"] = self.audio_prompt_path
            
            return self._model.generate(text, **kwargs)
        
        try:
            wav_tensor = await loop.run_in_executor(None, generate)
            
            # Convert to PCM at 24kHz
            pcm_data = wav_to_pcm_24k(wav_tensor, self._model_sr)
            
            # Yield in chunks for streaming compatibility
            for i in range(0, len(pcm_data), self.chunk_size):
                yield pcm_data[i:i + self.chunk_size]
            
            logger.debug(f"Chatterbox synthesis completed ({len(pcm_data)} bytes)")
            
        except Exception as e:
            logger.error(f"Chatterbox TTS error: {e}")
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


class ChatterboxMultilingualTTSProvider(BaseTTSProvider):
    """
    Multilingual TTS provider using Resemble AI's Chatterbox Multilingual model.
    
    Supports 23 languages: Arabic, Danish, German, Greek, English, Spanish,
    Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch,
    Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese.
    
    Example:
        provider = ChatterboxMultilingualTTSProvider(language="fr")
        await provider.initialize()
        async for audio_chunk in provider.synthesize_stream("Bonjour!"):
            pass
    """
    
    def __init__(
        self,
        language: ChatterboxLanguage = "en",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        audio_prompt_path: Optional[str] = None,
        chunk_size: int = 4096,
    ):
        """
        Initialize Chatterbox Multilingual TTS provider.
        
        Args:
            language: Language code (e.g., "en", "fr", "zh", "ja")
            device: Device to run on ("cpu", "cuda", or "auto")
            exaggeration: Emotion exaggeration control (0.0 to 1.0+)
            cfg_weight: Classifier-free guidance weight (0.0 to 1.0)
                       Set to 0 to avoid accent transfer from reference clip
            audio_prompt_path: Path to reference audio for voice cloning (optional)
            chunk_size: Size of audio chunks to yield (in bytes)
        """
        self.language = language
        self.device = device
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.audio_prompt_path = audio_prompt_path
        self.chunk_size = chunk_size
        
        self._model = None
        self._device = None
        self._sample_rate = 24000
    
    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._sample_rate
    
    async def initialize(self) -> None:
        """Load the Chatterbox Multilingual model."""
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError:
            raise ImportError(
                "chatterbox-tts not installed. Run: pip install chatterbox-tts"
            )
        
        # Determine device
        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device
        
        logger.info(f"Loading Chatterbox Multilingual TTS on {self._device}...")
        
        # Load model in thread pool
        loop = asyncio.get_event_loop()
        
        def load_model():
            return ChatterboxMultilingualTTS.from_pretrained(device=self._device)
        
        self._model = await loop.run_in_executor(None, load_model)
        
        # Store the model's native sample rate
        self._model_sr = self._model.sr
        
        logger.info(f"Chatterbox Multilingual TTS loaded (lang={self.language}, sr={self._model_sr})")
    
    async def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("Chatterbox Multilingual TTS unloaded")
    
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
        if self._model is None:
            await self.initialize()
        
        if not text or not text.strip():
            return
        
        logger.debug(f"Synthesizing [{self.language}]: {text[:50]}...")
        
        # Generate audio in thread pool
        loop = asyncio.get_event_loop()
        
        def generate():
            kwargs = {
                "language_id": self.language,
                "exaggeration": self.exaggeration,
                "cfg_weight": self.cfg_weight,
            }
            
            if self.audio_prompt_path:
                kwargs["audio_prompt_path"] = self.audio_prompt_path
            
            return self._model.generate(text, **kwargs)
        
        try:
            wav_tensor = await loop.run_in_executor(None, generate)
            
            # Convert to PCM at 24kHz
            pcm_data = wav_to_pcm_24k(wav_tensor, self._model_sr)
            
            # Yield in chunks for streaming compatibility
            for i in range(0, len(pcm_data), self.chunk_size):
                yield pcm_data[i:i + self.chunk_size]
            
            logger.debug(f"Multilingual synthesis completed ({len(pcm_data)} bytes)")
            
        except Exception as e:
            logger.error(f"Chatterbox Multilingual TTS error: {e}")
            raise
    
    async def synthesize(self, text: str) -> bytes:
        """Non-streaming synthesis."""
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)

