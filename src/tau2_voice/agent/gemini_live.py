"""
Gemini Live API Agent

Google Gemini Live API를 사용하는 음성 에이전트입니다.
- WebSocket 기반 실시간 양방향 통신
- 오디오 입력: 16-bit PCM, 16kHz, mono
- 오디오 출력: 24kHz
- Function calling 지원
- VAD (Voice Activity Detection) 지원

Required:
    pip install google-genai
"""
import asyncio
import base64
import json
from typing import Optional, Literal, AsyncGenerator, override

from loguru import logger

from tau2_voice.models.tool import Tool
from tau2_voice.agent.base import BaseAgent
from tau2_voice.models.events import (
    Event, AudioChunkEvent, AudioDoneEvent, TranscriptUpdateEvent,
    ToolCallRequestEvent, ToolCallResultEvent, SpeakRequestEvent
)
from tau2_voice.config import VoiceTauConfig

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not installed. Run: pip install google-genai")


def flatten_json_schema(schema: dict) -> dict:
    """
    Flatten JSON Schema by inlining $ref references and removing unsupported fields.
    Gemini API doesn't support $ref/$defs/additionalProperties in function parameters.
    """
    if not isinstance(schema, dict):
        return schema
    
    defs = schema.pop("$defs", {})
    
    # Fields not supported by Gemini Live API
    UNSUPPORTED_FIELDS = {
        "$defs", "$ref", "additionalProperties", "additional_properties",
        "title", "default", "$schema"
    }
    
    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                # Extract reference name from "#/$defs/TypeName"
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    type_name = ref_path.split("/")[-1]
                    if type_name in defs:
                        # Return a copy of the referenced definition
                        return resolve_refs(defs[type_name].copy())
                return {"type": "object"}  # Fallback
            
            # Remove unsupported fields and recurse
            return {k: resolve_refs(v) for k, v in obj.items() if k not in UNSUPPORTED_FIELDS}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj
    
    # Also handle anyOf with single $ref (common pattern)
    def simplify_anyof(obj):
        if isinstance(obj, dict):
            if "anyOf" in obj and len(obj["anyOf"]) == 1:
                # Replace anyOf with single item with the item itself
                return simplify_anyof(obj["anyOf"][0])
            if "items" in obj and isinstance(obj["items"], dict):
                if "anyOf" in obj["items"] and len(obj["items"]["anyOf"]) == 1:
                    obj["items"] = simplify_anyof(obj["items"]["anyOf"][0])
            return {k: simplify_anyof(v) for k, v in obj.items() if k not in UNSUPPORTED_FIELDS}
        elif isinstance(obj, list):
            return [simplify_anyof(item) for item in obj]
        return obj
    
    result = resolve_refs(schema)
    result = simplify_anyof(result)
    
    return result


class GeminiLiveAgent(BaseAgent):
    """
    Gemini Live API를 사용하는 실시간 음성 에이전트
    
    특징:
    - WebSocket 기반 양방향 스트리밍
    - Native audio output (자연스러운 음성)
    - VAD (Voice Activity Detection) 지원
    - Function calling 지원
    - Input/Output transcription 지원
    """
    
    def __init__(
        self,
        tools: Optional[list[Tool]],
        domain_policy: Optional[str],
        role: Literal["user", "assistant"] = "assistant",
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
        voice: str = "Kore",  # Available: Puck, Charon, Kore, Fenrir, Aoede
        sample_rate: int = 24000,
        use_vad: bool = True,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, role=role)
        
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        self.role = role
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        
        # Gemini client
        self._client = genai.Client(api_key=VoiceTauConfig.GOOGLE_API_KEY)
        self._session = None
        self._session_context = None  # Context manager for session
        self._is_connected = False
        
        # Event queue for async generator
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
        
        # Track current message for audio chunks
        self._current_message_id: Optional[str] = None
        self._message_counter = 0
        
        # Pending tool calls
        self._pending_tool_calls: dict = {}
    
    @property
    def system_prompt(self):
        return self.domain_policy or "You are a helpful assistant."
    
    def _build_config(self) -> dict:
        """Build Gemini Live API configuration"""
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self.system_prompt,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": self.voice
                    }
                }
            },
            # Enable transcription for both input and output
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            # VAD configuration - wait longer for speech to end
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
                    "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
                    "silence_duration_ms": 1000,  # Wait 1 second of silence before responding
                }
            }
        }
        
        # Add tools if available
        if self.tools:
            function_declarations = []
            for tool in self.tools:
                # Get description from tool's method
                description = ""
                if hasattr(tool, '_get_description'):
                    description = tool._get_description()
                elif hasattr(tool, 'short_desc'):
                    description = tool.short_desc
                
                func_decl = {
                    "name": tool.name,
                    "description": description,
                }
                
                # Get parameters schema and flatten it (remove $ref/$defs)
                if hasattr(tool, 'params'):
                    raw_schema = tool.params.model_json_schema()
                    func_decl["parameters"] = flatten_json_schema(raw_schema)
                
                function_declarations.append(func_decl)
            
            config["tools"] = [{"function_declarations": function_declarations}]
        
        return config
    
    @override
    async def connect(self):
        """Connect to Gemini Live API"""
        config = self._build_config()
        
        try:
            # Get the context manager and manually enter it
            self._session_context = self._client.aio.live.connect(
                model=self.model,
                config=config
            )
            self._session = await self._session_context.__aenter__()
            self._is_connected = True
            
            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            logger.info(f"[{self.role}] Connected to Gemini Live API (model={self.model}, voice={self.voice})")
        except Exception as e:
            logger.error(f"[{self.role}] Failed to connect to Gemini Live API: {e}")
            raise
    
    @override
    async def disconnect(self):
        """Disconnect from Gemini Live API"""
        self._is_connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        # Exit the context manager properly
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing Gemini session: {e}")
            self._session_context = None
            self._session = None
        
        logger.info(f"[{self.role}] Disconnected from Gemini Live API")
    
    @override
    async def publish(self, event: Event):
        """Publish event to Gemini Live API"""
        if not self._is_connected or self._session is None:
            return
        
        try:
            if event.type == "audio.chunk":
                # Send audio chunk to Gemini
                audio_data = base64.b64decode(event.audio_chunk)
                await self._session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )
                logger.debug(f"[{self.role}] Sent audio chunk ({len(audio_data)} bytes)")
            
            elif event.type == "audio.done":
                # Signal end of user's audio turn - Gemini needs this to start responding
                logger.debug(f"[{self.role}] Audio done received, sending turn_complete")
                # Use send_client_content with turn_complete to trigger Gemini response
                await self._session.send_client_content(
                    turns=None,  # No additional content, just signal turn complete
                    turn_complete=True
                )
            
            elif event.type == "speak.request":
                # For text-based requests (like after tool calls)
                if hasattr(event, 'instructions') and event.instructions:
                    await self._session.send_client_content(
                        turns={"role": "user", "parts": [{"text": event.instructions}]},
                        turn_complete=True
                    )
            
            elif event.type == "tool_call.result":
                # Send tool response back to Gemini
                function_response = types.FunctionResponse(
                    id=event.id,
                    name=self._pending_tool_calls.get(event.id, "unknown"),
                    response={"result": event.content}
                )
                await self._session.send_tool_response(
                    function_responses=[function_response]
                )
                # Clean up
                if event.id in self._pending_tool_calls:
                    del self._pending_tool_calls[event.id]
                    
        except Exception as e:
            logger.error(f"[{self.role}] Error publishing event: {e}")
    
    @override
    async def subscribe(self) -> AsyncGenerator[Event, None]:
        """Subscribe to events from Gemini Live API"""
        while self._is_connected:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    async def _receive_loop(self):
        """Background task to receive messages from Gemini"""
        if self._session is None:
            return
        
        try:
            async for response in self._session.receive():
                if not self._is_connected:
                    break
                
                await self._process_response(response)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.role}] Error in receive loop: {e}")
    
    async def _process_response(self, response):
        """Process a response from Gemini Live API"""
        try:
            # Debug: log response structure
            resp_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
            logger.debug(f"[{self.role}] Response attrs: {resp_attrs[:10]}")
            
            # Handle server content
            if hasattr(response, 'server_content') and response.server_content:
                server_content = response.server_content
                
                # Check for interruption
                if hasattr(server_content, 'interrupted') and server_content.interrupted:
                    logger.info(f"[{self.role}] Generation interrupted")
                    # Send audio done event to signal interruption
                    if self._current_message_id:
                        await self._event_queue.put(AudioDoneEvent(
                            role=self.role,
                            message_id=self._current_message_id
                        ))
                        self._current_message_id = None
                    return
                
                # Handle model turn (audio output)
                if hasattr(server_content, 'model_turn') and server_content.model_turn:
                    model_turn = server_content.model_turn
                    
                    for part in model_turn.parts:
                        # Handle audio data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if isinstance(part.inline_data.data, bytes):
                                # Generate message ID if needed
                                if self._current_message_id is None:
                                    self._message_counter += 1
                                    self._current_message_id = f"gemini_{self._message_counter}"
                                    logger.info(f"[{self.role}] Started new audio response: {self._current_message_id}")
                                
                                # Convert to base64 and send audio chunk
                                audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                await self._event_queue.put(AudioChunkEvent(
                                    role=self.role,
                                    message_id=self._current_message_id,
                                    audio_chunk=audio_b64
                                ))
                                logger.debug(f"[{self.role}] Sent audio output chunk ({len(part.inline_data.data)} bytes)")
                
                # Handle output transcription
                if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                    transcript = server_content.output_transcription.text
                    if transcript and self._current_message_id:
                        await self._event_queue.put(TranscriptUpdateEvent(
                            role=self.role,
                            message_id=self._current_message_id,
                            transcript=transcript
                        ))
                
                # Handle input transcription (from user audio)
                if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                    transcript = server_content.input_transcription.text
                    if transcript:
                        logger.debug(f"[{self.role}] Input transcription: {transcript}")
                
                # Handle turn complete
                if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                    logger.info(f"[{self.role}] Turn complete received")
                    if self._current_message_id:
                        await self._event_queue.put(AudioDoneEvent(
                            role=self.role,
                            message_id=self._current_message_id
                        ))
                        logger.info(f"[{self.role}] Sent AudioDoneEvent for {self._current_message_id}")
                        self._current_message_id = None
            
            # Handle tool calls
            if hasattr(response, 'tool_call') and response.tool_call:
                for fc in response.tool_call.function_calls:
                    # Store tool name for response
                    self._pending_tool_calls[fc.id] = fc.name
                    
                    # Parse arguments
                    args = fc.args if hasattr(fc, 'args') else {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    
                    # Emit tool call event
                    await self._event_queue.put(ToolCallRequestEvent(
                        event_id=f"event_{fc.id}",
                        message_id=f"msg_{fc.id}",
                        id=fc.id,
                        name=fc.name,
                        arguments=args
                    ))
                    
                    logger.info(f"[{self.role}] Tool call: {fc.name}({args})")
            
            # Handle usage metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                logger.debug(f"[{self.role}] Token usage: {usage.total_token_count}")
                
        except Exception as e:
            logger.error(f"[{self.role}] Error processing response: {e}")

