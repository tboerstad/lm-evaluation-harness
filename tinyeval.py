#!/usr/bin/env python3
"""
tinyeval - Minimal LLM evaluation harness.

Two tasks: gsm8k_llama (text) and chartqa (multimodal).
One abstraction: batch HTTP requests to an OpenAI-compatible API.
"""

from __future__ import annotations

import asyncio

from core import (
    APIConfig,
    _build_vision_message,
    _encode_image,
    _normalize,
    complete,
)
from tasks import TASKS

# Re-export core utilities for backwards compatibility
__all__ = [
    "APIConfig",
    "TASKS",
    "_build_vision_message",
    "_encode_image",
    "_normalize",
    "complete",
    "evaluate",
    "main",
]


async def evaluate(
    task_names: list[str], config: APIConfig, limit: int | None = None
) -> dict:
    """Run evaluations for specified tasks."""
    results = {}
    total_time = 0.0

    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
        result = await TASKS[name](config, limit)
        results[name] = result
        total_time += result["time_seconds"]

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

    task_names = [t.strip() for t in args.tasks.split(",")]
    output = asyncio.run(evaluate(task_names, config, args.limit))
    output["config"] = {"model": args.model, "limit": args.limit}

    print(json.dumps(output, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
