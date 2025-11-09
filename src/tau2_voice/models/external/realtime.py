from typing import Optional, Union
from pydantic import BaseModel


class RealtimeClientEvent(BaseModel):
    type: str
    event_id: Optional[str] = ""


# Conversation related models
class ConversationItemCreate(RealtimeClientEvent):
    type: str = "conversation.item.create"
    item: dict


# Input audio related models
class InputAudioBufferAppend(RealtimeClientEvent):
    type: str = "input_audio_buffer.append"
    audio: str  # base64 encoded audio data

class InputAudioBufferCommit(RealtimeClientEvent):
    type: str = "input_audio_buffer.commit"

class InputAudioBufferClear(RealtimeClientEvent):
    type: str = "input_audio_buffer.clear"


# Response related models
class Tool(BaseModel):
    type: str = "function"
    name: str
    description: str
    parameters: dict

class ResponseRequest(BaseModel):
    instructions: Optional[str] = None
    conversation: str = "auto"

class ResponseCreate(RealtimeClientEvent):
    type: str = "response.create"
    response: ResponseRequest = ResponseRequest()


# Session related models
class AudioFormat(BaseModel):
    type: str = "audio/pcm"
    rate: int = 24000

class NoiseReduction(BaseModel):
    type: str = "near_field"

class Transcription(BaseModel):
    language: str = "en"
    model: str = "gpt-4o-transcribe-latest"
    prompt: Optional[str] = None

class TurnDetection(BaseModel):
    type: str
    create_response: bool = True
    interrupt_response: bool = True

class ServerVAD(TurnDetection):
    type: str = "server_vad"
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    idle_timeout_ms: Optional[int] = None
    threshold: float = 0.5

class SemanticVAD(TurnDetection):
    type: str = "semantic_vad"
    eagerness: str = "high"

class AudioInput(BaseModel):
    format: AudioFormat
    noise_reduction: NoiseReduction
    turn_detection: TurnDetection

class AudioOutput(BaseModel):
    format: AudioFormat
    speed: float = 1.0
    voice: str

class Audio(BaseModel):
    input: AudioInput
    output: AudioOutput

class Session(BaseModel):
    type: str = "realtime"
    model: str = "gpt-realtime"
    instructions: str
    tools: list[Tool] = []
    tool_choice: str = "auto"
    max_output_tokens: Union[int, str] = "inf"
    output_modalities: list[str] = ["audio"]
    audio: Optional[Audio] = None
    include: Optional[list[str]] = None
    
class SessionUpdate(RealtimeClientEvent):
    type: str = "session.update"
    session: Session
