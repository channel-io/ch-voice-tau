# Event Adapter for Qwen3-Omni API

import base64
import struct
import io
import json
from typing import Literal
from loguru import logger

from tau2_voice.adapters.base import BaseEventAdapter
from tau2_voice.models.events import (
    Event, AudioChunkEvent, AudioDoneEvent, 
    TranscriptUpdateEvent, ToolCallRequestEvent
)


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    PCM raw 데이터를 WAV 형식으로 변환 (헤더 추가)
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    
    # WAV 헤더 생성
    wav_header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',                    # ChunkID
        36 + data_size,             # ChunkSize
        b'WAVE',                    # Format
        b'fmt ',                    # Subchunk1ID
        16,                         # Subchunk1Size (PCM)
        1,                          # AudioFormat (PCM = 1)
        channels,                   # NumChannels
        sample_rate,                # SampleRate
        byte_rate,                  # ByteRate
        block_align,                # BlockAlign
        bits_per_sample,            # BitsPerSample
        b'data',                    # Subchunk2ID
        data_size                   # Subchunk2Size
    )
    
    return wav_header + pcm_data


class Qwen3OmniEventAdapter(BaseEventAdapter):
    """
    Qwen3-Omni는 turn-based 시스템이므로 오디오 청크를 누적하다가
    turn end 시그널을 받으면 전체 오디오를 API로 전송합니다.
    """
    def __init__(self, role: Literal["user", "assistant"]):
        self.role = role
        self.audio_buffer = []  # 오디오 청크를 누적할 버퍼
        
    def wrap_event(self, event: Event) -> dict | None:
        """tau2 event를 Qwen3-Omni API 형식으로 변환"""
        match event.type:
            case "audio.chunk":
                # 오디오 청크를 버퍼에 누적
                self.audio_buffer.append(event.audio_chunk)
                return None  # 아직 전송하지 않음
                
            case "audio.done":
                # turn이 끝났으므로 누적된 오디오를 메시지로 구성
                if not self.audio_buffer:
                    return None
                    
                # 모든 청크를 합침 (base64 인코딩된 상태)
                combined_b64 = "".join(self.audio_buffer)
                self.audio_buffer = []  # 버퍼 초기화
                
                # base64 디코딩하여 raw PCM 데이터 얻기
                pcm_data = base64.b64decode(combined_b64)
                
                # PCM을 WAV로 변환 (24kHz, 16-bit, mono)
                wav_data = pcm_to_wav(pcm_data, sample_rate=24000, channels=1, bits_per_sample=16)
                
                # WAV 데이터를 base64로 인코딩
                wav_b64 = base64.b64encode(wav_data).decode('utf-8')
                
                # data URL 형식으로 변환
                audio_data_url = f"data:audio/wav;base64,{wav_b64}"
                
                logger.debug(f"Audio converted: PCM {len(pcm_data)} bytes -> WAV {len(wav_data)} bytes")
                
                return {
                    "audio_url": audio_data_url,
                    "message_id": event.message_id
                }
                
            case "tool_call.result":
                # tool call 결과를 API 형식으로 변환
                return {
                    "type": "tool_result",
                    "call_id": event.id,
                    "content": event.content
                }
            
            case _:
                return None
    
    def unwrap_event(self, response: dict) -> list[Event]:
        """
        Qwen3-Omni API 응답을 tau2 event들로 변환
        
        response 형식:
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "텍스트 응답",
                    "tool_calls": [...]
                }
            }]
        }
        """
        events = []
        
        if "choices" not in response or not response["choices"]:
            return events
            
        choice = response["choices"][0]
        message = choice.get("message", {})
        message_id = response.get("id", "unknown")
        
        # 텍스트 응답을 transcript로 변환
        content = message.get("content", "")
        if content:
            events.append(TranscriptUpdateEvent(
                role=self.role,
                event_id=f"{message_id}_transcript",
                message_id=message_id,
                transcript=content
            ))
        
        # tool calls 처리
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                func = tool_call.get("function", {})
                # arguments가 문자열이면 JSON 파싱
                arguments = func.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments: {arguments}")
                        arguments = {}
                events.append(ToolCallRequestEvent(
                    role=self.role,
                    event_id=f"{message_id}_{tool_call.get('id', 'unknown')}",
                    message_id=message_id,
                    id=tool_call.get("id", "unknown"),
                    name=func.get("name", ""),
                    arguments=arguments,
                    requestor=self.role
                ))
        
        return events

