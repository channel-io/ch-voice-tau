import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

from tau2_voice.agent import BaseAgent
from tau2_voice.models.events import (
    Event,
    AudioChunkEvent,
    AudioDoneEvent,
    TranscriptUpdateEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
    ConversationEndEvent,
)
from tau2_voice.models.events import SpeakRequestEvent
from tau2_voice.audio.collector import AudioCollector
from tau2_voice.utils.utils import get_now

from tau2.environment.environment import Environment
from tau2.data_model.tasks import Task
from tau2.data_model.message import AssistantMessage, UserMessage, ToolCall, ToolMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType


DEFAULT_FIRST_INSTRUCTIONS = """⚠️ SPEAK IN ENGLISH ONLY! NO Arabic, Spanish, Korean, Chinese, or any other language! ⚠️

YOU ARE THE CUSTOMER seeking help.

CRITICAL: 
1. SPEAK IN ENGLISH
2. Read your scenario carefully 
3. Start the conversation about YOUR SPECIFIC PROBLEM mentioned in the scenario

DO NOT:
- Speak any language other than ENGLISH
- Ask "What can I help you with"
- Talk about random devices or things not in your scenario
- Offer to help anyone

DO:
- SPEAK IN ENGLISH
- Mention your ORDER NUMBER or ACCOUNT issue from your scenario
- Ask for help with YOUR SPECIFIC problem IN ENGLISH"""

class VoiceOrchestrator:
    def __init__(
        self,
        domain: str,
        assistant: BaseAgent,
        user: BaseAgent,
        environment: Environment,
        task: Task,
    ):
        self.domain = domain
        self.assistant = assistant
        self.user = user
        self.environment = environment
        self.task = task
        self._task_agent: asyncio.Task | None = None
        self._task_user: asyncio.Task | None = None
        
        # Track messages for evaluation
        self.messages: list = []
        self.start_time: str | None = None
        self.start_perf: float | None = None
        self.termination_reason: TerminationReason = TerminationReason.AGENT_STOP
        self.turn_idx = 0
        
        # Track pending tool calls to merge with transcript messages
        self.pending_tool_calls: dict = {}  # message_id -> tool_calls

        # Initialize audio collector
        timestamp = get_now()
        out_dir = Path("data/recordings") / self.domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.domain}_{getattr(self.task, 'id', 'task')}_{timestamp}.wav"
        self.collector = AudioCollector(output_wav_path=out_path)

    async def run(self) -> SimulationRun:
        self.start_time = get_now()
        self.start_perf = time.perf_counter()
        
        # Initialize environment
        if self.task.initial_state:
            initialization_data = self.task.initial_state.initialization_data
            initialization_actions = self.task.initial_state.initialization_actions
            self.environment.set_state(
                initialization_data=initialization_data,
                initialization_actions=initialization_actions,
                message_history=[],
            )
        self.environment.sync_tools()
        
        await self._connect_participants()
        try:
            await self._start_event_loops()
        finally:
            simulation_run = await self._shutdown()
        
        # Evaluate the simulation
        try:
            reward_info = evaluate_simulation(
                simulation=simulation_run,
                task=self.task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=self.domain,
            )
            simulation_run.reward_info = reward_info
            
            # Calculate success and finalize audio collector with metadata
            reward = reward_info.reward if reward_info else 0.0
            success = reward >= 1.0
            self.collector.finalize(success=success, reward=reward)
            
            # Log results
            logger.info(f"Simulation completed")
            logger.info(f"Termination reason: {simulation_run.termination_reason}")
            logger.info(f"Duration: {simulation_run.duration:.2f}s")
            logger.info(f"Reward: {reward}")
            if reward_info.db_check:
                logger.info(f"DB Match: {reward_info.db_check.db_match}")
                logger.info(f"DB Reward: {reward_info.db_check.db_reward}")
            if reward_info.reward_breakdown:
                logger.info(f"Reward breakdown: {reward_info.reward_breakdown}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            logger.error(f"Message count: {len(simulation_run.messages)}")
            # Finalize collector even on error (without success/reward)
            try:
                self.collector.finalize()
            except:
                pass
            raise
        
        return simulation_run

    async def _connect_participants(self):
        await asyncio.gather(self.assistant.connect(), self.user.connect())
        logger.info("Participants connected: assistant and user")

    async def _start_event_loops(self):
        self._task_agent = asyncio.create_task(
            self._run_subscription_loop(self.assistant, target=self.user)
        )
        self._task_user = asyncio.create_task(
            self._run_subscription_loop(self.user, target=self.assistant)
        )
        # Kick off with user - remind them of their role and scenario
        scenario_str = str(self.task.user_scenario) if hasattr(self.task, 'user_scenario') else 'Check your instructions'
        
        # Log the scenario being used
        logger.info(f"\n{'='*80}")
        logger.info(f"TASK SCENARIO for Task {self.task.id}:")
        logger.info(f"{'='*80}")
        logger.info(scenario_str[:500])  # Log first 500 chars
        logger.info(f"{'='*80}\n")
        
        first_instructions = f"""{DEFAULT_FIRST_INSTRUCTIONS}

IMPORTANT: Follow your scenario below. Start the conversation by mentioning your specific problem from the scenario.

Your scenario: {scenario_str}"""
        
        await self.user.publish(SpeakRequestEvent(instructions=first_instructions))
        try:
            await asyncio.gather(self._task_agent, self._task_user)
        except asyncio.CancelledError:
            pass

    async def _shutdown(self) -> SimulationRun:
        tasks = [t for t in [self._task_agent, self._task_user] if t is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Note: collector.finalize() is called in run() after evaluation
        # to include success and reward information
        
        await asyncio.gather(self.assistant.disconnect(), self.user.disconnect())
        logger.info("Participants disconnected")
        
        # Debug: Log message sequence
        logger.debug(f"\nMessage sequence for evaluation ({len(self.messages)} messages):")
        for i, msg in enumerate(self.messages):
            msg_type = type(msg).__name__
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                logger.debug(f"  {i}: {msg_type} with {len(msg.tool_calls)} tool_calls")
            elif hasattr(msg, 'id') and msg.role == 'tool':
                logger.debug(f"  {i}: {msg_type} (id={msg.id})")
            else:
                content_preview = getattr(msg, 'content', '')[:50] if hasattr(msg, 'content') else ''
                logger.debug(f"  {i}: {msg_type} - {content_preview}...")
        
        # Create SimulationRun
        duration = time.perf_counter() - self.start_perf if self.start_perf else 0
        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=self.start_time or get_now(),
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason,
            reward_info=None,
            agent_cost=None,
            user_cost=None,
            messages=self.messages,
            seed=None,
        )
        return simulation_run

    async def _run_subscription_loop(self, source: BaseAgent, target: BaseAgent):
        try:
            async for event in source.subscribe():
                await self._route_event(event, source=source, target=target)
        except asyncio.CancelledError:
            pass

    async def _route_event(self, event: Event, source: BaseAgent, target: BaseAgent):
        match event.type:
            case "audio.chunk":
                await self._handle_audio_chunk(event, target)
            case "audio.done":
                await self._handle_audio_done(event, target)
            case "transcript.update":
                await self._handle_transcript_update(event, source)
            case "tool_call.request":
                await self._handle_tool_call_request(event, source)

    async def _handle_audio_chunk(self, event: AudioChunkEvent, target: BaseAgent):
        # record chunk
        try:
            self.collector.handle_audio_chunk(role=event.role, message_id=event.message_id, audio_chunk_b64=event.audio_chunk)
        except Exception:
            pass
        await target.publish(event)

    async def _handle_audio_done(self, event: AudioDoneEvent, target: BaseAgent):
        # close current turn
        try:
            self.collector.handle_audio_done(role=event.role, message_id=event.message_id)
        except Exception:
            pass
        # Forward audio.done to target first (needed for turn-based agents like Qwen3-Omni)
        await target.publish(event)
        # Then send speak request to trigger response generation
        await target.publish(SpeakRequestEvent())

    async def _handle_transcript_update(self, event: TranscriptUpdateEvent, source: BaseAgent):
        logger.info(f"[{source.role}] Transcript update: {event.transcript}")
        
        # Store transcript in audio collector
        try:
            self.collector.handle_transcript_update(
                role=event.role,
                message_id=event.message_id,
                transcript=event.transcript,
            )
        except Exception as e:
            logger.warning(f"Failed to store transcript in collector: {e}")
        
        transcript_upper = event.transcript.upper().strip()
        
        # Check for special termination tokens from user
        if source.role == "user":
            if "###TRANSFER###" in transcript_upper:
                logger.info("User generated ###TRANSFER### token - ending conversation")
                self.termination_reason = TerminationReason.USER_STOP
                if self._task_agent:
                    self._task_agent.cancel()
                if self._task_user:
                    self._task_user.cancel()
                return
            elif "###OUT-OF-SCOPE###" in transcript_upper:
                logger.info("User generated ###OUT-OF-SCOPE### token - ending conversation")
                self.termination_reason = TerminationReason.USER_STOP
                if self._task_agent:
                    self._task_agent.cancel()
                if self._task_user:
                    self._task_user.cancel()
                return
        
        # Check for transfer message from assistant
        if source.role == "assistant":
            if "YOU ARE BEING TRANSFERRED" in transcript_upper or "BEING TRANSFERRED TO A HUMAN" in transcript_upper:
                logger.info("Assistant sent transfer message - ending conversation after this message")
                # Let this message be recorded, then end the conversation
                timestamp = get_now()
                msg = AssistantMessage(
                    role="assistant",
                    content=event.transcript,
                    timestamp=timestamp,
                    turn_idx=self.turn_idx,
                    cost=0.0,
                )
                self.messages.append(msg)
                self.turn_idx += 1
                
                # End the conversation
                self.termination_reason = TerminationReason.AGENT_STOP
                if self._task_agent:
                    self._task_agent.cancel()
                if self._task_user:
                    self._task_user.cancel()
                return
        
        # Convert to Message for evaluation
        timestamp = get_now()
        if source.role == "assistant":
            msg = AssistantMessage(
                role="assistant",
                content=event.transcript,
                timestamp=timestamp,
                turn_idx=self.turn_idx,
                cost=0.0,
            )
        else:  # user
            msg = UserMessage(
                role="user",
                content=event.transcript,
                timestamp=timestamp,
                turn_idx=self.turn_idx,
                cost=0.0,
            )
        self.messages.append(msg)
        self.turn_idx += 1

    async def _handle_tool_call_request(self, event: ToolCallRequestEvent, source: BaseAgent):
        # Check if this is a stop_conversation call from user
        if event.name == "stop_conversation" and source.role == "user":
            logger.info("User called stop_conversation - ending conversation")
            # Send a simple confirmation result
            result_event = ToolCallResultEvent(
                event_id=event.event_id,
                message_id=event.message_id,
                id=event.id,
                content="Task completed. Ending conversation.",
                success=True,
            )
            await source.publish(result_event)
            # Cancel the event loops to end the conversation
            if self._task_agent:
                self._task_agent.cancel()
            if self._task_user:
                self._task_user.cancel()
            return
        
        # Handle regular tool calls
        tool_message = self.environment.get_response(event)
        logger.info(f"Tool call request: {event}")
        logger.info(f"Tool call response: {tool_message}")
        
        # Debug: Log which agent made the call and their tools
        agent = source
        logger.info(f"[DEBUG] Tool '{event.name}' called by: {agent.role}")
        logger.info(f"[DEBUG] {agent.role} has {len(agent.tools)} tools: {[t.name for t in agent.tools]}")
        logger.info(f"[DEBUG] {agent.role} system prompt (first 500 chars): {agent.system_prompt[:500] if hasattr(agent, 'system_prompt') else 'N/A'}")
        
        # Create tool call
        tool_call = ToolCall(
            id=event.id,
            name=event.name,
            arguments=event.arguments,
        )
        
        # Check if we need to merge with a previous assistant message
        # Look for the most recent AssistantMessage without tool_calls
        merged = False
        assistant_msg_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if isinstance(msg, AssistantMessage):
                # If this assistant message has no tool calls yet, add the tool call to it
                if not msg.tool_calls:
                    msg.tool_calls = [tool_call]
                    logger.debug(f"Merged tool call into existing AssistantMessage at index {i}")
                    merged = True
                    assistant_msg_idx = i
                break
        
        # If not merged, create a new AssistantMessage
        if not merged:
            timestamp = get_now()
            assistant_msg = AssistantMessage(
                role="assistant",
                content="",
                tool_calls=[tool_call],
                timestamp=timestamp,
                turn_idx=self.turn_idx,
                cost=0.0,
            )
            self.messages.append(assistant_msg)
            assistant_msg_idx = len(self.messages) - 1
            self.turn_idx += 1
            logger.debug(f"Created new AssistantMessage with tool call at index {assistant_msg_idx}")
        
        # Add tool response message RIGHT AFTER the assistant message with tool call
        # This ensures the order: AssistantMessage with tool_calls -> ToolMessage
        tool_response = ToolMessage(
            role="tool",
            id=tool_message.id,
            content=tool_message.content,
            error=not tool_message.success if hasattr(tool_message, 'success') else False,
            requestor="assistant",
            timestamp=get_now(),
            turn_idx=self.turn_idx,
        )
        # Insert right after the assistant message
        self.messages.insert(assistant_msg_idx + 1, tool_response)
        self.turn_idx += 1
        
        result_event = ToolCallResultEvent(
            event_id=event.event_id,
            message_id=event.message_id,
            **tool_message.model_dump(),
        )
        await source.publish(result_event)
        # After sending tool result, trigger response generation
        await source.publish(SpeakRequestEvent())