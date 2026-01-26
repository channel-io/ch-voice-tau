from tau2_voice.agent.base import BaseAgent
from tau2_voice.agent.realtime import RealtimeAgent
from tau2_voice.agent.user import UserAgent
from tau2_voice.agent.qwen3_omni import Qwen3OmniAgent
from tau2_voice.agent.gemini_live import GeminiLiveAgent
from tau2_voice.agent.cascade import CascadeAgent

# HumanAgent requires sounddevice/PortAudio which may not be available on servers
# Import lazily to avoid import errors
try:
    from tau2_voice.agent.human import HumanAgent
except (ImportError, OSError):
    HumanAgent = None  # type: ignore

__all__ = [
    "BaseAgent",
    "RealtimeAgent",
    "HumanAgent",
    "UserAgent",
    "Qwen3OmniAgent",
    "GeminiLiveAgent",
    "CascadeAgent",
]
