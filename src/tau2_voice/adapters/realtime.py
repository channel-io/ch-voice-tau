# Event Adapter for OpenAI Realtime API

import json
from typing import Literal
from json import JSONDecodeError
from loguru import logger

from tau2_voice.adapters.base import BaseEventAdapter
from tau2_voice.models.events import (
    Event, AudioChunkEvent, AudioDoneEvent, 
    TranscriptUpdateEvent, ToolCallRequestEvent
)
from tau2_voice.models.external.realtime import (
    RealtimeClientEvent, InputAudioBufferAppend,
    ConversationItemCreate, ResponseCreate, ResponseRequest
)


class RealtimeEventAdapter(BaseEventAdapter):
    def __init__(self, role: Literal["user", "assistant"]):
        self.role = role
    
    def wrap_event(self, event: Event) -> RealtimeClientEvent | None:
        match event.type:
            case "audio.chunk":
                # NOTE: OpenAI Realtime input audio is always treated as "user" audio by the API.
                # For the *user simulator* session (self.role == "user"), assistant audio MUST NOT
                # be fed as input audio, or the simulator will transcribe the assistant's words as
                # if they were spoken by the user and start responding like an assistant.
                if self.role == "assistant":
                    return InputAudioBufferAppend(
                        audio=event.audio_chunk
                    )
                return None
            case "transcript.update":
                # For the user simulator, inject the assistant's transcript as a conversation item
                # (role="assistant") so the simulator can react as the customer.
                if self.role == "user" and getattr(event, "role", None) == "assistant":
                    item = {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "input_text", "text": event.transcript}],
                    }
                    logger.info(
                        f"Sending assistant transcript to user simulator via conversation item: {event.transcript[:120]}"
                    )
                    return ConversationItemCreate(item=item)
                return None
            case "speak.request":
                return ResponseCreate(
                    response=ResponseRequest(
                        instructions=event.instructions,
                    )
                )
            case "tool_call.result":
                item = {
                    "type": "function_call_output",
                    "call_id": event.id,
                    "output": event.content
                }
                logger.info(f"Sending conversation item create: {item}")
                return ConversationItemCreate(
                    item=item
                )
            
    def unwrap_event(self, event: dict) -> Event | None:
        match event["type"]:
            case "response.output_audio.delta":
                return AudioChunkEvent(
                    role=self.role,
                    event_id=event["event_id"],
                    message_id=event["item_id"],
                    audio_chunk=event["delta"]
                )
            case "response.output_audio.done":
                return AudioDoneEvent(
                    role=self.role,
                    event_id=event["event_id"],
                    message_id=event["item_id"]
                )
            case "response.output_audio_transcript.done":
                return TranscriptUpdateEvent(
                    role=self.role,
                    event_id=event["event_id"],
                    message_id=event["item_id"],
                    transcript=event["transcript"]
                )
            case "response.output_item.done":
                if event["item"]["type"] == "function_call":
                    try:
                        arguments = json.loads(event["item"]["arguments"])
                    except JSONDecodeError:
                        return Event(
                            role=self.role,
                            event_id=event["event_id"],
                            error=f"Failed to parse arguments: {event['item']['arguments']}"
                        )
                    return ToolCallRequestEvent(
                        role=self.role,
                        event_id=event["event_id"],
                        message_id=event["item"]["id"],
                        id=event["item"]["call_id"],
                        name=event["item"]["name"],
                        arguments=arguments,
                        requestor=self.role  # CRITICAL: Set requestor to match who is calling
                    )
                else:
                    return None
            case "error":
                return Event(
                    role=self.role,
                    event_id=event["event_id"],
                    error_message=event["error"]["message"],
                )
            case _:
                return None