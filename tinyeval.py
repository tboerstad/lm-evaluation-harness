#!/usr/bin/env python3
"""
tinyeval CLI entry point.

Responsibilities:
- Parse CLI args (tasks, model, endpoint, concurrency)
- Create APIConfig, run tasks
- Output JSON

Architecture:
    tinyeval.py (CLI, orchestration)
         │
    ┌────┴────┐
  core.py   tasks/
  APIConfig   TASKS registry
  complete()  gsm8k.py
  run_task()  chartqa.py

Flow: CLI → APIConfig → evaluate() → TASKS[name]() → JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json

from core import APIConfig, TaskResult, run_task
from tasks import TASKS


def _parse_gen_kwargs(s: str) -> dict:
    """Parse 'key=value,key=value' into dict with type inference."""
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        # Type inference
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
        elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            result[key] = int(value)
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


async def evaluate(
    task_names: list[str], config: APIConfig, max_samples: int | None = None
) -> dict[str, TaskResult | float]:
    """Run evaluations for specified tasks."""
    results: dict[str, TaskResult] = {}
    total_seconds = 0.0

    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
        result = await run_task(TASKS[name], config, max_samples)
        results[name] = result
        total_seconds += result["elapsed"]

    return {"results": results, "total_seconds": round(total_seconds, 2)}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="tinyeval - Minimal LLM evaluation")
    parser.add_argument(
        "--tasks", required=True, help=f"Comma-separated: {', '.join(TASKS.keys())}"
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--base_url", required=True, help="API base URL")
    parser.add_argument("--api_key", default="", help="API key")
    parser.add_argument("--max_samples", type=int, help="Max samples per task")
    parser.add_argument(
        "--num_concurrent", type=int, default=8, help="Concurrent requests"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help="Generation kwargs: key=value,key=value (e.g. temperature=0.7,max_tokens=1024)",
    )
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    config = APIConfig(
        url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        num_concurrent=args.num_concurrent,
        seed=args.seed,
        gen_kwargs=_parse_gen_kwargs(args.gen_kwargs),
    )

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output = asyncio.run(evaluate(task_names, config, args.max_samples))
    output["config"] = {"model": args.model, "max_samples": args.max_samples}

    print(json.dumps(output, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
