"""
Benchmark script for gpt-realtime-1.5 across all domains.
Run: OPENAI_API_KEY=<key> uv run python scripts/run_realtime_1_5_benchmark.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tau2_voice.run import run_all_tasks
from loguru import logger


ASSISTANT_MODEL = "gpt-realtime-1.5"
USER_MODEL = "gpt-realtime-2025-08-28"
NUM_TASKS = 50
BATCH_SIZE = 5


async def run_domain(domain: str):
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING BENCHMARK: {domain} with {ASSISTANT_MODEL}")
    logger.info(f"{'='*80}\n")

    results, accuracy = await run_all_tasks(
        domain=domain,
        num_tasks=NUM_TASKS,
        batch_size=BATCH_SIZE,
        assistant_model=ASSISTANT_MODEL,
        user_model=USER_MODEL,
    )

    return {
        "domain": domain,
        "model": ASSISTANT_MODEL,
        "accuracy": accuracy,
        "num_tasks": len(results),
        "successful": sum(1 for r in results if r.get("success", False)),
        "results": results,
    }


async def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else None

    if domain:
        # Run single domain
        result = await run_domain(domain)
        output_file = Path(f"data/benchmark_realtime_1_5_{domain}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"\n{domain}: {result['accuracy']:.1%} ({result['successful']}/{result['num_tasks']})")
    else:
        # Run all domains
        all_results = {}
        for d in ["retail", "airline", "telecom"]:
            result = await run_domain(d)
            all_results[d] = result
            output_file = Path(f"data/benchmark_realtime_1_5_{d}.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY: gpt-realtime-1.5")
        logger.info(f"{'='*80}")
        for d, r in all_results.items():
            logger.info(f"  {d}: {r['accuracy']:.1%} ({r['successful']}/{r['num_tasks']})")


if __name__ == "__main__":
    asyncio.run(main())
