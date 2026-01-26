"""
Providers for the cascade pipeline.

This module contains pluggable provider implementations for:
- ASR (Automatic Speech Recognition)
- LLM (Large Language Model)
- TTS (Text-to-Speech)

Usage:
    from tau2_voice.providers.asr import WhisperLocalProvider
    from tau2_voice.providers.llm import OpenAILLMProvider, LocalLLMProvider
    from tau2_voice.providers.tts import OpenAITTSProvider
"""

from tau2_voice.providers.base import (
    BaseASRProvider,
    BaseLLMProvider,
    BaseTTSProvider,
    LLMResponse,
)

__all__ = [
    "BaseASRProvider",
    "BaseLLMProvider",
    "BaseTTSProvider",
    "LLMResponse",
]
