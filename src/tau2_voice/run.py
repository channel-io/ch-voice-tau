import asyncio
from pathlib import Path

from typing import Optional
from loguru import logger
from tqdm import tqdm

from tau2_voice.models.tasks import Task
from tau2_voice.models.tool import Tool
from tau2_voice.registry import registry
from tau2_voice.orchestrator.orchestrator import VoiceOrchestrator
from tau2_voice.utils.stop_tool import stop_conversation_tool

# Cascade agent imports
from tau2_voice.agent.cascade import CascadeAgent
from tau2_voice.providers.asr import WhisperLocalProvider, OpenAIASRProvider
from tau2_voice.providers.llm import OpenAILLMProvider, LocalLLMProvider
from tau2_voice.providers.tts import OpenAITTSProvider, ChatterboxTTSProvider


def load_tasks(task_set_name: str) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    global registry
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader()
    return tasks


def create_cascade_agent(
    tools: list[Tool],
    domain_policy: str,
    asr_model: str = "openai",
    llm_model: str = "gpt-4o",
    tts_model: str = "openai",
) -> CascadeAgent:
    """
    Create a CascadeAgent with explicitly configured providers.
    
    Args:
        tools: Available tools for the agent
        domain_policy: System prompt / policy for the LLM
        asr_model: ASR model - "openai", "whisper-large-v3", "whisper-base", etc.
        llm_model: LLM model - "gpt-4o", "gpt-4o-mini", "nemotron"
        tts_model: TTS model - "openai", "chatterbox"
        
    Returns:
        Configured CascadeAgent instance
    """
    # Create ASR provider
    if asr_model == "openai":
        asr_provider = OpenAIASRProvider(language="en")
        logger.info("Using OpenAI ASR (Whisper API)")
    else:
        # Map short names to full HuggingFace model IDs
        whisper_model_map = {
            "whisper-large-v3": "openai/whisper-large-v3",
            "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
            "whisper-large-v2": "openai/whisper-large-v2",
            "whisper-large": "openai/whisper-large",
            "whisper-medium": "openai/whisper-medium",
            "whisper-small": "openai/whisper-small",
            "whisper-base": "openai/whisper-base",
            "whisper-tiny": "openai/whisper-tiny",
        }
        model_id = whisper_model_map.get(asr_model, f"openai/{asr_model}")
        asr_provider = WhisperLocalProvider(
            model_id=model_id,
            device="auto",
            language="en",
        )
        logger.info(f"Using local Whisper ASR: {model_id}")
    
    # Create LLM provider
    if llm_model == "nemotron":
        llm_provider = LocalLLMProvider(
            model_id="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
            device="auto",
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            thinking="off",
        )
        logger.info("Using local LLM: Nemotron")
    else:
        llm_provider = OpenAILLMProvider(
            model=llm_model,
            temperature=0.7,
        )
        logger.info(f"Using OpenAI LLM: {llm_model}")
    
    # Create TTS provider
    if tts_model == "chatterbox":
        tts_provider = ChatterboxTTSProvider(
            device="auto",
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        logger.info("Using Chatterbox TTS (local)")
    else:
        tts_provider = OpenAITTSProvider(
            model="gpt-4o-mini-tts",
            voice="alloy",
        )
        logger.info("Using OpenAI TTS")
    
    logger.info(f"Creating CascadeAgent: ASR={asr_model}, LLM={llm_model}, TTS={tts_model}")
    
    return CascadeAgent(
        tools=tools,
        domain_policy=domain_policy,
        asr_provider=asr_provider,
        llm_provider=llm_provider,
        tts_provider=tts_provider,
        role="assistant",
    )


def get_tasks(
    task_set_name: str,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    if task_ids is None:
        tasks = load_tasks(task_set_name=task_set_name)
    else:
        tasks = [
            task for task in load_tasks(task_set_name=task_set_name) if task.id in task_ids
        ]
    if task_ids is not None and len(tasks) != len(task_ids):
        missing_tasks = set(task_ids) - set([task.id for task in tasks])
        raise ValueError(
            f"Not all tasks were found for task set {task_set_name}: {missing_tasks}"
        )
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


async def run_task(
    domain: str,
    task: Task,
    assistant_model: str = "gpt-4o-realtime-preview-2024-12-17",
    user_model: str = "gpt-4o-realtime-preview-2024-12-17",
    asr_model: str = "openai",
    llm_model: str = "gpt-4o",
    tts_model: str = "openai",
):
    global registry

    EnvironmentConstructor = registry.get_env_constructor(domain)
    environment = EnvironmentConstructor()

    # Get user tools from environment (e.g., check_status_bar for telecom)
    # Some domains (airline, retail) don't have user tools
    try:
        user_tools = environment.get_user_tools() if hasattr(environment, 'get_user_tools') else []
    except Exception as e:
        logger.warning(f"No user tools available for domain '{domain}': {e}")
        user_tools = []
    
    # Add the stop_conversation tool for ending the conversation
    user_tools.append(stop_conversation_tool)
    
    logger.info(f"User tools ({len(user_tools)}): {[tool.name for tool in user_tools]}")

    UserConstructor = registry.get_agent_constructor("user_agent")
    user = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        model=user_model,
    )

    assistant_tools = environment.get_tools()
    logger.info(f"Assistant tools: {[tool.name for tool in assistant_tools]}")
    
    # Select agent based on model name
    if assistant_model == "cascade":
        # Create cascade agent with configurable providers
        assistant = create_cascade_agent(
            tools=assistant_tools,
            domain_policy=environment.get_policy(),
            asr_model=asr_model,
            llm_model=llm_model,
            tts_model=tts_model,
        )
    elif assistant_model.startswith("qwen3"):
        agent_name = "qwen3_omni_agent"
        AssistantConstructor = registry.get_agent_constructor(agent_name)
        assistant = AssistantConstructor(
            tools=assistant_tools,
            domain_policy=environment.get_policy(),
            role="assistant",
            model=assistant_model,
        )
    elif assistant_model.startswith("gemini"):
        agent_name = "gemini_live_agent"
        AssistantConstructor = registry.get_agent_constructor(agent_name)
        assistant = AssistantConstructor(
            tools=assistant_tools,
            domain_policy=environment.get_policy(),
            role="assistant",
            model=assistant_model,
        )
    else:
        agent_name = "realtime_agent"
        AssistantConstructor = registry.get_agent_constructor(agent_name)
        assistant = AssistantConstructor(
            tools=assistant_tools,
            domain_policy=environment.get_policy(),
            role="assistant",
            model=assistant_model,
        )
    
    orchestrator = VoiceOrchestrator(
        domain=domain,
        assistant=assistant,
        user=user,
        environment=environment,
        task=task,
    )

    logger.info(f"Running task {task.id} for domain {domain}")  
    simulation_run = await orchestrator.run()
    
    logger.info(f"Task {task.id} completed with reward: {simulation_run.reward_info.reward if simulation_run.reward_info else 'N/A'}")
    
    return simulation_run


async def run_task_with_index(
    domain: str, 
    task: Task, 
    task_idx: int, 
    total: int, 
    assistant_model: str, 
    user_model: str,
    asr_model: str = "openai",
    llm_model: str = "gpt-4o",
    tts_model: str = "openai",
):
    """Run a single task and return indexed result."""
    # try:
    logger.info(f"[{task_idx+1}/{total}] Starting task: {task.id}")
    simulation_run = await run_task(
        domain, 
        task, 
        assistant_model=assistant_model, 
        user_model=user_model,
        asr_model=asr_model,
        llm_model=llm_model,
        tts_model=tts_model,
    )
    
    reward = simulation_run.reward_info.reward if simulation_run.reward_info else 0.0
    is_success = reward >= 1.0
    
    logger.info(f"[{task_idx+1}/{total}] Task {task.id} completed: {'✓' if is_success else '✗'} (reward={reward:.3f})")
    
    return {
        'task_id': task.id,
        'simulation_id': simulation_run.id,
        'reward': reward,
        'success': is_success,
        'duration': simulation_run.duration,
        'index': task_idx,
    }
    # except Exception as e:
    #     logger.error(f"[{task_idx+1}/{total}] Error running task {task.id}: {e}")
    #     return {
    #         'task_id': task.id,
    #         'simulation_id': None,
    #         'reward': 0.0,
    #         'success': False,
    #         'duration': 0.0,
    #         'error': str(e),
    #         'index': task_idx,
    #     }


async def run_all_tasks(
    domain: str, 
    task_ids: Optional[list[str]] = None, 
    num_tasks: Optional[int] = None, 
    batch_size: int = 10,
    assistant_model: str = "gpt-4o-realtime-preview-2024-12-17",
    user_model: str = "gpt-4o-realtime-preview-2024-12-17",
    asr_model: str = "openai",
    llm_model: str = "gpt-4o",
    tts_model: str = "openai",
):
    """
    Run all tasks in parallel batches.
    
    Args:
        domain: Domain name
        task_ids: Optional list of task IDs to run
        num_tasks: Optional number of tasks to run
        batch_size: Number of tasks to run in parallel (default: 10)
        assistant_model: Model to use for assistant (default: gpt-4o-realtime-preview-2024-12-17)
        user_model: Model to use for user simulator (default: gpt-4o-realtime-preview-2024-12-17)
        asr_model: ASR model for cascade (default: openai)
        llm_model: LLM model for cascade (default: gpt-4o)
        tts_model: TTS model for cascade (default: openai)
    """
    tasks = get_tasks(domain, task_ids=task_ids, num_tasks=num_tasks)
    total_tasks = len(tasks)
    
    logger.info(f"Starting evaluation on {total_tasks} tasks from domain '{domain}'")
    logger.info(f"Assistant model: {assistant_model}")
    logger.info(f"User model: {user_model}")
    logger.info(f"Running with batch size: {batch_size}")
    logger.info("=" * 80)
    
    all_results = []
    
    # Use tqdm for progress bar
    pbar = tqdm(total=total_tasks, desc="Running tasks", unit="task")
    
    # Process tasks in batches with real-time progress updates
    for batch_start in range(0, total_tasks, batch_size):
        batch_end = min(batch_start + batch_size, total_tasks)
        batch_tasks = tasks[batch_start:batch_end]
        
        logger.info(f"\nProcessing batch {batch_start//batch_size + 1}: tasks {batch_start+1}-{batch_end}")
        
        # Create tasks but don't wait for all at once
        pending = [
            asyncio.create_task(run_task_with_index(
                domain, task, batch_start + i, total_tasks, 
                assistant_model, user_model,
                asr_model, llm_model, tts_model,
            ))
            for i, task in enumerate(batch_tasks)
        ]
        
        # Update progress bar as tasks complete
        for coro in asyncio.as_completed(pending):
            result = await coro
            all_results.append(result)
            pbar.update(1)
            
            # Update progress bar with current stats
            successful_tasks = sum(1 for r in all_results if r.get('success', False))
            current_accuracy = successful_tasks / len(all_results)
            pbar.set_postfix({
                'Success': f"{successful_tasks}/{len(all_results)}",
                'Accuracy': f"{current_accuracy:.3f}"
            })
    
    pbar.close()
    
    # Sort results by original index
    all_results.sort(key=lambda x: x.get('index', 0))
    results = all_results
    
    # Final report
    final_accuracy = successful_tasks / total_tasks
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Domain: {domain}")
    logger.info(f"Total Tasks: {total_tasks}")
    logger.info(f"Successful: {successful_tasks}")
    logger.info(f"Failed: {total_tasks - successful_tasks}")
    logger.info(f"Final Accuracy: {successful_tasks}/{total_tasks} = {final_accuracy:.3f}")
    logger.info(f"{'='*80}\n")
    
    # Print summary table
    logger.info("\nTask Summary:")
    logger.info("-" * 80)
    logger.info(f"{'Task ID':<10} {'Success':<10} {'Reward':<10} {'Duration (s)':<15}")
    logger.info("-" * 80)
    for result in results:
        status = '✓' if result['success'] else '✗'
        logger.info(f"{result['task_id']:<10} {status:<10} {result['reward']:<10.3f} {result['duration']:<15.2f}")
    logger.info("-" * 80)
    
    return results, final_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run τ-bench voice evaluation")
    parser.add_argument("--domain", default="airline", choices=["airline", "retail", "telecom"])
    parser.add_argument("--assistant-model", default="cascade", 
                       help="Model: cascade, gpt-4o-realtime-*, gemini-*, qwen3_omni")
    parser.add_argument("--user-model", default="gpt-realtime-2025-08-28")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Specific task IDs to run")
    parser.add_argument("--num-tasks", type=int, default=None, help="Number of tasks to run")
    parser.add_argument("--batch-size", type=int, default=1, help="Parallel batch size (use 1 for local models)")
    
    # Cascade-specific model options
    parser.add_argument("--asr-model", default="openai",
                       help="ASR model for cascade: openai, whisper-large-v3, whisper-base, etc.")
    parser.add_argument("--llm-model", default="gpt-4o",
                       help="LLM model for cascade: gpt-4o, gpt-4o-mini, nemotron")
    parser.add_argument("--tts-model", default="openai",
                       help="TTS model for cascade: openai, chatterbox")
    
    args = parser.parse_args()
    
    # Model selection reference:
    # - "gpt-4o-realtime-preview-2024-12-17" - OpenAI Realtime API
    # - "gpt-realtime-mini-2025-10-06" - OpenAI Realtime Mini
    # - "gemini-2.5-flash-native-audio-preview-12-2025" - Gemini Live
    # - "qwen3_omni" - Qwen3-Omni (requires local vLLM server)
    # 
    # Cascade pipeline (--assistant-model cascade):
    #   --asr-model: openai (default), whisper-large-v3, whisper-base, whisper-small, whisper-medium
    #   --llm-model: gpt-4o (default), gpt-4o-mini, nemotron (local)
    #   --tts-model: openai (default), chatterbox (local)
    
    # Default to single task if nothing specified
    task_ids = args.task_ids or ["0"]
    
    logger.info(f"Running evaluation:")
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Assistant: {args.assistant_model}")
    if args.assistant_model == "cascade":
        logger.info(f"    ASR: {args.asr_model}")
        logger.info(f"    LLM: {args.llm_model}")
        logger.info(f"    TTS: {args.tts_model}")
    logger.info(f"  User: {args.user_model}")
    logger.info(f"  Tasks: {task_ids if args.task_ids else f'first {args.num_tasks or 1}'}")
    
    results, accuracy = asyncio.run(run_all_tasks(
        domain=args.domain,
        task_ids=task_ids if args.task_ids else None,
        num_tasks=args.num_tasks or (1 if not args.task_ids else None),
        batch_size=args.batch_size,
        assistant_model=args.assistant_model,
        user_model=args.user_model,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
    ))
    
    logger.info(f"\nEvaluation completed with final accuracy: {accuracy:.3f}")