#!/usr/bin/env python3
"""
tinyeval - Minimal LLM evaluation harness.

Two tasks: gsm8k_llama (text) and chartqa (multimodal).
One abstraction: batch HTTP requests to an OpenAI-compatible API.

Control Inversion: This module runs the evaluation loop.
Task modules (gsm8k, chartqa) are passive libraries that provide:
- load(limit) -> list
- prompt(doc) -> str | tuple
- score(responses, docs) -> dict
- STOP -> list[str] | None
"""

from __future__ import annotations

import asyncio
import time

from core import APIConfig, complete
from tasks import TASKS


async def evaluate(
    task_names: list[str], config: APIConfig, limit: int | None = None
) -> dict:
    """Run evaluations for specified tasks."""
    results = {}
    total_time = 0.0

    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")

        task = TASKS[name]

        # 1. Load data
        docs = task.load(limit)
        print(f"Evaluating: {name} ({len(docs)} samples)")

        # 2. Prepare prompts
        prompts = [task.prompt(d) for d in docs]

        # 3. Run inference (centralized timing)
        t0 = time.perf_counter()
        responses = await complete(prompts, config, stop=task.STOP)
        elapsed = time.perf_counter() - t0

        # 4. Score
        metrics = task.score(responses, docs)

        # 5. Report
        print(f"{name}: {metrics} ({elapsed:.2f}s)")
        results[name] = {
            "task": name,
            "metrics": metrics,
            "num_samples": len(docs),
            "time_seconds": round(elapsed, 2),
        }
        total_time += elapsed

    return {"results": results, "total_time_seconds": round(total_time, 2)}


def main() -> int:
    """CLI entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="tinyeval - Minimal LLM evaluation")
    parser.add_argument(
        "--tasks", required=True, help=f"Comma-separated: {', '.join(TASKS.keys())}"
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--base_url", required=True, help="API base URL")
    parser.add_argument("--api_key", default="", help="API key")
    parser.add_argument("--limit", type=int, help="Limit samples per task")
    parser.add_argument(
        "--num_concurrent", type=int, default=8, help="Concurrent requests"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    config = APIConfig(
        url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        num_concurrent=args.num_concurrent,
        seed=args.seed,
    )

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output = asyncio.run(evaluate(task_names, config, args.limit))
    output["config"] = {"model": args.model, "limit": args.limit}

    print(json.dumps(output, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
