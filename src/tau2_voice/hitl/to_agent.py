import asyncio
from loguru import logger

from tau2.registry import registry
from tau2_voice.models.tool import Tool
from tau2_voice.agent import Agent, RealtimeAgent, HumanAgent
from tau2_voice.models.events import (
    Event, Message, AudioChunkEvent, AudioDoneEvent, 
    TranscriptUpdateEvent, ToolCallRequestEvent, ToolCallResultEvent
)


async def main(domain: str):
    global registry
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()

    assistant = RealtimeAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        role="assistant",
    )
    user = HumanAgent(
        tools=[],
        domain_policy="",
        role="user",
    )

    async def run_assistant():
        await assistant.connect()
        try:
            async for event in assistant.subscribe():
                await handle_event(event, user)
        except asyncio.CancelledError:
            pass
    
    async def run_user():
        await user.connect()
        try:
            async for event in user.subscribe():
                await handle_event(event, assistant)
        except asyncio.CancelledError:
            pass

    async def handle_event(
        event: Event,
        target: Agent,
    ) -> Message | None:
        match event.type:
            case "audio.chunk":
                await _handle_audio_chunk(event, target)
            case "audio.done":
                await _handle_audio_done(event, target)
            case "transcript.update":
                await _handle_transcript_update(event, target)
            case "tool_call.request":
                await _handle_tool_call_request(event, target)

    async def _handle_audio_chunk(event: AudioChunkEvent, target: Agent):
        # Always publish the event to the target; each agent handles playback/input appropriately
        await target.publish(event)

    async def _handle_audio_done(event: AudioDoneEvent, target: Agent):
        logger.info(f"Audio done")

    async def _handle_transcript_update(event: TranscriptUpdateEvent, target: Agent):
        logger.info(f"Transcript update: {event.transcript}")

    async def _handle_tool_call_request(event: ToolCallRequestEvent, target: Agent):
        tool_message = environment.get_response(event)
        logger.info(f"Tool call request: {event}")
        logger.info(f"Tool call request: {tool_message}")
        tool_result_event = ToolCallResultEvent(
            event_id=event.event_id,
            message_id=event.message_id,
            **tool_message.model_dump(),
        )
        # Tool results should go back to the assistant (LLM) side
        await assistant.publish(tool_result_event)

    task_agent = asyncio.create_task(run_assistant())
    task_user = asyncio.create_task(run_user())
    try:
        await asyncio.gather(task_agent, task_user)
    except asyncio.CancelledError:
        pass
    finally:
        task_agent.cancel()
        task_user.cancel()
        await asyncio.gather(task_agent, task_user, return_exceptions=True)
        await assistant.disconnect()
        await user.disconnect()
    

if __name__ == "__main__":
    asyncio.run(main("retail"))