"""
launch vllm server:
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8901 --host 127.0.0.1 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 4 --enable-auto-tool-choice --tool-call-parser hermes  
"""
import asyncio
import json
import base64
from typing import Optional, Literal, AsyncGenerator, override

import aiohttp
from loguru import logger
from openai import AsyncOpenAI

from tau2_voice.models.tool import Tool
from tau2_voice.agent.base import BaseAgent
from tau2_voice.models.events import Event, AudioChunkEvent, AudioDoneEvent, TranscriptUpdateEvent
from tau2_voice.adapters.qwen3_omni import Qwen3OmniEventAdapter
from tau2_voice.config import VoiceTauConfig


class Qwen3OmniAgent(BaseAgent):
    """
    Qwen3-Omni는 turn-based 시스템입니다:
    1. 오디오 청크를 누적
    2. turn end 시그널 받으면 HTTP API로 전송
    3. 텍스트 응답을 gpt-4o-mini-tts로 변환
    4. audio chunk로 스트리밍
    """
    
    def __init__(
        self,
        tools: Optional[list[Tool]],
        domain_policy: Optional[str],
        role: Literal["user", "assistant"] = "assistant",
        model: str = "qwen3-omni",
        api_base: str = "http://localhost:8901/v1",
        tts_model: str = "gpt-4o-mini-tts",
        tts_voice: str = "alloy",
        sample_rate: int = 24000,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, role=role)
        self.role = role
        # Map model name to actual API model name
        if "qwen3" in model.lower():
            self.model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        else:
            self.model = model
        self.api_base = api_base
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.sample_rate = sample_rate
        self.adapter = Qwen3OmniEventAdapter(role=self.role)
        
        # HTTP client
        self._session: Optional[aiohttp.ClientSession] = None
        
        # OpenAI TTS client
        self._openai = AsyncOpenAI(api_key=VoiceTauConfig.OPENAI_API_KEY)
        
        # Event queue for async generator
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._messages: list[dict] = []
        self._is_connected = False
        self._pending_tool_response = False  # tool 결과 후 응답 대기 플래그
    
    @property
    def system_prompt(self):
        return self.domain_policy or "You are a helpful assistant."
    
    @override
    async def connect(self):
        """HTTP 세션 초기화"""
        self._session = aiohttp.ClientSession()
        self._is_connected = True
        
        # 시스템 메시지 추가
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        logger.info(f"[{self.role}] Qwen3-Omni agent connected")
    
    @override
    async def disconnect(self):
        """리소스 정리"""
        self._is_connected = False
        if self._session:
            await self._session.close()
            self._session = None
        logger.info(f"[{self.role}] Qwen3-Omni agent disconnected")
    
    @override
    async def publish(self, event: Event):
        """
        이벤트 발행:
        - audio.chunk: 버퍼에 누적
        - audio.done: API 호출하고 TTS로 변환하여 응답
        - speak.request: tool 결과 후 응답 생성
        - tool_call.result: 메시지에 추가하고 플래그 설정
        """
        if not self._is_connected:
            return
        
        # audio.done 이벤트일 때만 실제로 API 호출
        if event.type == "audio.done":
            await self._process_turn()
        elif event.type == "audio.chunk":
            # adapter가 내부적으로 버퍼에 누적
            self.adapter.wrap_event(event)
        elif event.type == "speak.request":
            # tool call 결과 후에는 텍스트만으로 응답 생성
            if self._pending_tool_response:
                self._pending_tool_response = False
                await self._process_turn_after_tool()
        elif event.type == "tool_call.result":
            # tool 결과를 메시지에 추가
            tool_result = self.adapter.wrap_event(event)
            if tool_result:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["call_id"],
                    "content": tool_result["content"]
                })
                self._pending_tool_response = True
    
    @override
    async def subscribe(self) -> AsyncGenerator[Event, None]:
        """이벤트 구독 - queue에서 이벤트를 가져옴"""
        while self._is_connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue
    
    async def _process_turn(self):
        """
        turn이 끝났을 때 처리:
        1. 누적된 오디오를 Qwen3-Omni API로 전송
        2. 텍스트 응답을 TTS로 변환
        3. audio chunk로 스트리밍
        """
        # 1. 오디오 메시지 구성
        audio_data = self.adapter.wrap_event(
            AudioDoneEvent(role=self.role, message_id="current_turn")
        )
        
        if not audio_data:
            logger.warning("No audio data to process")
            return
        
        # 메시지에 오디오 추가
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_data["audio_url"]}},
            ]
        })
        
        # 2. Qwen3-Omni API 호출
        try:
            logger.debug("Calling Qwen3-Omni API...")
            response = await self._call_qwen3_omni()
            logger.debug(f"API response received, choices: {len(response.get('choices', []))}")
            
            # 3. 응답 파싱
            events = self.adapter.unwrap_event(response)
            logger.debug(f"Parsed {len(events)} events from response")
            
            # transcript 추출
            transcript = ""
            for event in events:
                if isinstance(event, TranscriptUpdateEvent):
                    transcript = event.transcript
                    await self._event_queue.put(event)
                elif event.type == "tool_call.request":
                    await self._event_queue.put(event)
            
            # 4. TTS로 변환하여 audio chunk 스트리밍
            if transcript:
                # Limit transcript length to avoid very long TTS
                if len(transcript) > 500:
                    logger.warning(f"Transcript too long ({len(transcript)} chars), truncating to 500")
                    transcript = transcript[:500] + "..."
                await self._stream_tts(transcript)
            else:
                logger.warning("No transcript in response, sending empty audio done")
                # Send audio done even if no transcript to continue conversation
                done_event = AudioDoneEvent(role=self.role, message_id="empty_response")
                await self._event_queue.put(done_event)
            
            # 5. assistant 메시지를 히스토리에 추가
            if "choices" in response and response["choices"]:
                self._messages.append(response["choices"][0]["message"])
            
            logger.debug("Turn processing completed")
                
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_turn_after_tool(self):
        """
        Tool call 결과 후 응답 생성:
        오디오 없이 텍스트 응답만 생성
        """
        try:
            logger.debug("Processing turn after tool call result")
            
            # Qwen3-Omni API 호출
            response = await self._call_qwen3_omni()
            logger.debug(f"API response received after tool, choices: {len(response.get('choices', []))}")
            
            # 응답 파싱
            events = self.adapter.unwrap_event(response)
            logger.debug(f"Parsed {len(events)} events from response")
            
            # transcript 추출
            transcript = ""
            for event in events:
                if isinstance(event, TranscriptUpdateEvent):
                    transcript = event.transcript
                    await self._event_queue.put(event)
                elif event.type == "tool_call.request":
                    await self._event_queue.put(event)
            
            # TTS로 변환하여 audio chunk 스트리밍
            if transcript:
                # Limit transcript length to avoid very long TTS
                if len(transcript) > 500:
                    logger.warning(f"Transcript too long ({len(transcript)} chars), truncating to 500")
                    transcript = transcript[:500] + "..."
                await self._stream_tts(transcript)
            else:
                logger.warning("No transcript in response after tool, sending empty audio done")
                done_event = AudioDoneEvent(role=self.role, message_id="empty_response")
                await self._event_queue.put(done_event)
            
            # assistant 메시지를 히스토리에 추가
            if "choices" in response and response["choices"]:
                self._messages.append(response["choices"][0]["message"])
            
            logger.debug("Turn after tool processing completed")
                
        except Exception as e:
            logger.error(f"Error processing turn after tool: {e}")
            import traceback
            traceback.print_exc()

    async def _call_qwen3_omni(self) -> dict:
        """Qwen3-Omni API 호출"""
        if not self._session:
            raise RuntimeError("Session not initialized")
        
        # tools를 OpenAI 형식으로 변환
        tools_schema = None
        if self.tools:
            tools_schema = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.short_desc or tool.name,
                        "parameters": tool.params.model_json_schema(),
                    }
                }
                for tool in self.tools
            ]
        
        payload = {
            "model": self.model,
            "messages": self._messages,
        }
        
        if tools_schema:
            payload["tools"] = tools_schema
        
        logger.debug(f"Calling Qwen3-Omni API with {len(self._messages)} messages")
        
        async with self._session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"API error {resp.status}: {error_text}")
            
            return await resp.json()
    
    async def _stream_tts(self, text: str):
        """
        텍스트를 gpt-4o-mini-tts로 변환하여 audio chunk로 스트리밍
        """
        message_id = f"tts_{hash(text)}"
        chunks_sent = 0
        
        try:
            logger.info(f"Generating TTS for: {text[:50]}...")
            
            # TTS 스트리밍 (instructions는 간단하게)
            async with self._openai.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text,
                instructions="Speak clearly and naturally.",  # 가벼운 instruction
                response_format="pcm",
            ) as response:
                # PCM 데이터를 chunk로 읽어서 전송
                chunk_size = 4096  # 바이트 단위
                
                try:
                    async for chunk in response.iter_bytes(chunk_size):
                        if not self._is_connected:
                            break
                        
                        # base64 인코딩
                        audio_b64 = base64.b64encode(chunk).decode('utf-8')
                        
                        # audio chunk 이벤트 생성
                        event = AudioChunkEvent(
                            role=self.role,
                            message_id=message_id,
                            audio_chunk=audio_b64
                        )
                        
                        await self._event_queue.put(event)
                        chunks_sent += 1
                except Exception as stream_error:
                    # 스트리밍 중 에러 발생해도 이미 받은 청크는 유지
                    logger.warning(f"TTS streaming interrupted after {chunks_sent} chunks: {stream_error}")
                
                logger.info(f"TTS streaming completed for message {message_id} ({chunks_sent} chunks)")
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            # 항상 audio done 이벤트 전송 (부분적 오디오라도 완료 처리)
            done_event = AudioDoneEvent(
                role=self.role,
                message_id=message_id
            )
            await self._event_queue.put(done_event)

