import asyncio
import ssl
import json
import contextlib
import websockets

from loguru import logger
from typing import Optional, Literal, AsyncGenerator, override

from tau2_voice.models.tool import Tool

from tau2_voice.agent.base import BaseAgent
from tau2_voice.models.events import Event
from tau2_voice.adapters.realtime import RealtimeEventAdapter
from tau2_voice.config import VoiceTauConfig
from tau2_voice.models.external.realtime import (
    ResponseCreate, Session, RealtimeClientEvent, Audio, AudioInput, AudioFormat, NoiseReduction, 
    AudioOutput, SessionUpdate, SemanticVAD, Tool as OpenAITool
)


class RealtimeAgent(BaseAgent):
    def __init__(
        self,
        tools: Optional[list[Tool]],
        domain_policy: Optional[str],
        role: Literal["user", "assistant"] = "assistant",
        model: str = "gpt-realtime",
        voice: str = "alloy",
        sample_rate: int = 24000,
        use_vad: bool = True
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, role=role)
        self.role = role
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        self.adapter = RealtimeEventAdapter(role=self.role)
        self._ws = None
        self._task: Optional[asyncio.Task] = None
        self._sink = None
    
    @property
    def url(self):
        return f"wss://api.openai.com/v1/realtime?model={self.model}"

    @property
    def system_prompt(self):
        return self.domain_policy

    @override
    async def connect(self):
        headers = {
            "Authorization": f"Bearer {VoiceTauConfig.OPENAI_API_KEY}",
        }
        self._ws = await websockets.connect(
            self.url,
            additional_headers=headers,
            max_size=None,
            ssl=ssl.create_default_context(),
            ping_interval=20,
            ping_timeout=20,
        )
        await self.update_session()
        logger.info("Connected to realtime agent")

    @override
    async def disconnect(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._sink:
            await self._sink.close()

    @override
    async def publish(self, event: Event):
        if self._ws is None:
            return
        wrapped_event = self.adapter.wrap_event(event)
        if wrapped_event is not None:
            await self._send_event(wrapped_event)

    @override
    async def subscribe(self) -> AsyncGenerator[Event, None]:
        if self._ws is None:
            return
        async for raw in self._ws:
            event = json.loads(raw)
            event = self.adapter.unwrap_event(event)
            if event is None:
                continue
            yield event

    async def _send_event(self, event: RealtimeClientEvent):
        assert self._ws is not None
        await self._ws.send(event.model_dump_json(exclude_none=True))

    async def update_session(self):
        openai_tools = [tool.openai_schema for tool in self.tools]
        realtime_tools = [OpenAITool(
            type="function",
            name=tool["function"]["name"],
            description=tool["function"]["description"],
            parameters=tool["function"]["parameters"],
        ) for tool in openai_tools]
        
        logger.info(f"[{self.role}] Registering {len(realtime_tools)} tools with Realtime API: {[t.name for t in realtime_tools]}")
        logger.info(f"[{self.role}] Session instructions (first 200 chars): {self.system_prompt if self.system_prompt else 'None'}...")
        
        session = Session(
            instructions=self.system_prompt,
            tools=realtime_tools,
            model=self.model,
            audio=Audio(
                input=AudioInput(
                    format=AudioFormat(type="audio/pcm", rate=self.sample_rate),
                    noise_reduction=NoiseReduction(type="near_field"),
                    turn_detection=SemanticVAD() if self.use_vad else None,
                ),
                output=AudioOutput(
                    format=AudioFormat(type="audio/pcm", rate=self.sample_rate),
                    voice=self.voice,
                ),
            ),
            output_modalities=["audio"],
        )
        await self._send_event(SessionUpdate(session=session))
