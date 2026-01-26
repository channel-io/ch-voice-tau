"""ASR (Automatic Speech Recognition) providers."""

from tau2_voice.providers.asr.whisper_local import WhisperLocalProvider
from tau2_voice.providers.asr.openai import OpenAIASRProvider

__all__ = ["WhisperLocalProvider", "OpenAIASRProvider"]
