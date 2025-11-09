import base64

from loguru import logger
from typing import Literal, AsyncGenerator, override, Optional

from tau2_voice.models.tool import Tool
from tau2_voice.agent.base import BaseAgent
from tau2_voice.models.events import Event, AudioChunkEvent
from tau2_voice.audio.source import MicrophoneSource
from tau2_voice.audio.sink import SpeakerSink


class HumanAgent(BaseAgent):
    def __init__(
        self,
        tools: Optional[list[Tool]],
        domain_policy: Optional[str],
        role: Literal["user", "assistant"] = "assistant",
    ):
        logger.info(f"You are now a human agent. Use your microphone to communicate.")
        logger.info(f"Domain policy: {domain_policy}")
        logger.info(f"Tools: {tools}")
        super().__init__(tools=tools, domain_policy=domain_policy, role=role)
        self.source = None
        self.sink = None

    @override
    async def connect(self):
        self.source = MicrophoneSource()
        self.sink = SpeakerSink()
        logger.info("Connected to human agent")
        
    @override
    async def disconnect(self):
        if self.source:
            await self.source.close()

    @override
    async def publish(self, event: Event):
        if event.type == "audio.chunk":
            assert self.sink is not None
            await self.sink.write(base64.b64decode(event.audio_chunk))
        elif event.type == "tool_call.result":
            print(f"Tool call result: {event.content}")

    @override
    async def subscribe(self) -> AsyncGenerator[Event]:
        async for chunk in self.source.read():
            event = AudioChunkEvent(
                role=self.role,
                audio_chunk=base64.b64encode(chunk).decode()
            )
            yield event
