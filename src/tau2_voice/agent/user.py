import asyncio
import ssl
import json
import contextlib
import websockets

from loguru import logger
from typing import Optional, Literal, AsyncGenerator, override

from tau2_voice.models.tool import Tool

from tau2_voice.agent.realtime import RealtimeAgent
from tau2_voice.adapters.realtime import RealtimeEventAdapter
from tau2_voice.audio.sink import SpeakerSink
from tau2_voice.config import VoiceTauConfig
from tau2_voice.models.events import Event
from tau2_voice.models.tasks import UserInstructions
from tau2_voice.models.external.realtime import (
    Session, RealtimeClientEvent, Audio, AudioInput, AudioFormat, NoiseReduction, 
    ServerVAD, AudioOutput, SessionUpdate, Tool as OpenAITool, SemanticVAD
)

GLOBAL_USER_SIM_GUIDELINES_DIR = VoiceTauConfig.DATA_DIR / "tau2" / "user_simulator"


GLOBAL_USER_SIM_GUIDELINES_PATH = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_tools.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_VOICE = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_voice.md"
)

def get_global_user_sim_guidelines(use_tools: bool = False) -> str:
    """
    Get the global user simulator guidelines.

    Args:
        use_tools: Whether to use the tools guidelines.

    Returns:
        The global user simulator guidelines.
    """
    if use_tools:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS, "r") as fp:
            user_sim_guidelines = fp.read()
    else:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH_VOICE, "r") as fp:
            user_sim_guidelines = fp.read()
    return user_sim_guidelines

SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()


class UserAgent(RealtimeAgent):
    def __init__(
        self,
        tools: Optional[list[Tool]],
        instructions: Optional[UserInstructions],
        voice: str = "ash",
        **kwargs,
    ):
        # IMPORTANT: pass through kwargs (e.g. model=...) so callers can control the
        # underlying Realtime model used for the user simulator.
        super().__init__(tools=tools, domain_policy=None, role="user", voice=voice, **kwargs)
        self.instructions = instructions
        
    @property
    def global_simulation_guidelines(self) -> str:
        """
        The simulation guidelines for the user simulator.
        """
        use_tools = bool(self.tools)
        return get_global_user_sim_guidelines(use_tools=use_tools)
    
    @property
    def system_prompt(self):
        if self.instructions is None:
            logger.warning("No instructions provided for user agent")
        system_prompt = SYSTEM_PROMPT.format(
            global_user_sim_guidelines=self.global_simulation_guidelines,
            instructions=self.instructions,
        )
        logger.debug(f"User system prompt: {system_prompt}")
        return system_prompt
