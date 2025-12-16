import asyncio
from pathlib import Path

from typing import Optional
from loguru import logger
from tqdm import tqdm

from tau2_voice.models.tasks import Task
from tau2_voice.registry import registry
from tau2_voice.orchestrator.orchestrator import VoiceOrchestrator
from tau2_voice.utils.stop_tool import stop_conversation_tool


def load_tasks(task_set_name: str) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    global registry
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader()
    return tasks


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
    if assistant_model.startswith("qwen3"):
        agent_name = "qwen3_omni_agent"
    elif assistant_model.startswith("gemini"):
        agent_name = "gemini_live_agent"
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


async def run_task_with_index(domain: str, task: Task, task_idx: int, total: int, assistant_model: str, user_model: str):
    """Run a single task and return indexed result."""
    try:
        logger.info(f"[{task_idx+1}/{total}] Starting task: {task.id}")
        simulation_run = await run_task(domain, task, assistant_model=assistant_model, user_model=user_model)
        
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
    except Exception as e:
        logger.error(f"[{task_idx+1}/{total}] Error running task {task.id}: {e}")
        return {
            'task_id': task.id,
            'simulation_id': None,
            'reward': 0.0,
            'success': False,
            'duration': 0.0,
            'error': str(e),
            'index': task_idx,
        }


async def run_all_tasks(
    domain: str, 
    task_ids: Optional[list[str]] = None, 
    num_tasks: Optional[int] = None, 
    batch_size: int = 10,
    assistant_model: str = "gpt-4o-realtime-preview-2024-12-17",
    user_model: str = "gpt-4o-realtime-preview-2024-12-17",
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
            asyncio.create_task(run_task_with_index(domain, task, batch_start + i, total_tasks, assistant_model, user_model))
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
    domain = "airline" # airline, retail, telecom
    
    # Model selection
    # assistant_model = "gpt-realtime-mini-2025-10-06"  # or "gpt-realtime-mini-2025-10-06"
    assistant_model = "qwen3_omni"
    user_model = "gpt-realtime-2025-08-28"
    
    # Run all tasks (or specify task_ids or num_tasks)
    num_tasks = 50  # Run first task only
    task_ids = None  # Set to None to use num_tasks, or specify IDs for airline/retail
    
    # For telecom, use num_tasks since task IDs are complex strings
    # For airline/retail, you can use task_ids like ["0", "1", "2"]
    
    results, accuracy = asyncio.run(run_all_tasks(
        domain=domain,
        task_ids=["7"],
        num_tasks=num_tasks,
        batch_size=5,  # Run 10 tasks in parallel
        assistant_model=assistant_model,
        user_model=user_model,
    ))
    
    logger.info(f"\nEvaluation completed with final accuracy: {accuracy:.3f}")