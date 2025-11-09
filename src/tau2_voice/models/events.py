import uuid
from typing import Literal, Optional
from pydantic import BaseModel, Field
from tau2_voice.models.message import ToolCall, ToolMessage


class Event(BaseModel):
    """
    Base event class for all events.
    """
    event_id: str = Field(description="The unique identifier of the event.", default_factory=uuid.uuid4)
    type: str = Field(description="The type of the event.", default="event")
    error_message: Optional[str] = Field(description="The error message.", default=None)


class SpeakRequestEvent(Event):
    """
    A speak request event.
    """
    type: str = "speak.request"
    message: Optional[str] = Field(description="The message to speak.", default=None)
    instructions: Optional[str] = Field(description="The instructions to speak.", default=None)


class MessageEvent(Event):
    """
    Base message event class for all message events.
    """
    type: str = "message"
    role: Literal["assistant", "user", "tool"] = "assistant"
    message_id: str = Field(description="The unique identifier of the message.", default_factory=uuid.uuid4)


class AudioChunkEvent(MessageEvent):
    """
    An audio chunk event.
    """
    type: str = "audio.chunk"
    audio_chunk: str = Field(description="The base64 encoded audio chunk.")


class AudioDoneEvent(MessageEvent):
    """
    An audio done event.
    """
    type: str = "audio.done"


class TranscriptUpdateEvent(MessageEvent):
    """
    A transcript update event.
    """
    type: str = "transcript.update"
    transcript: str = Field(description="The transcript of the audio chunk.")


class ToolCallRequestEvent(MessageEvent, ToolCall):
    type: str = "tool_call.request"


class ToolCallResultEvent(MessageEvent, ToolMessage):
    type: str = "tool_call.result"


class ConversationEndEvent(Event):
    """
    A conversation end event to signal the conversation should terminate.
    """
    type: str = "conversation.end"
    reason: str = Field(description="The reason for ending the conversation.", default="task_completed")
    

class Message(BaseModel):
    """
    A message in the conversation.
    """
    index: int
    role: Literal["assistant", "user"]
    transcript: str
    message_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_results: Optional[list[ToolMessage]] = None