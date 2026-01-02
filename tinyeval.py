#!/usr/bin/env python3
"""tinyeval CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any, TypedDict

from core import APIConfig, TaskResult, run_task
from tasks import TASKS


class EvalResult(TypedDict):
    results: dict[str, TaskResult]
    total_seconds: float
    config: dict[str, Any]


def _parse_kwargs(s: str) -> dict[str, Any]:
    """Parse 'key=value,key=value' into dict."""
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid format '{pair}'")
        k, v = pair.split("=", 1)
        result[k] = json.loads(v)
    return result


async def evaluate(
    task_names: list[str], config: APIConfig, max_samples: int | None = None
) -> EvalResult:
    """Run evaluations for specified tasks."""
    results, total = {}, 0.0
    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
        results[name] = await run_task(TASKS[name], config, max_samples)
        total += results[name]["elapsed"]
    return {
        "results": results,
        "total_seconds": round(total, 2),
        "config": {"model": config.model, "max_samples": max_samples},
    }


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="tinyeval - Minimal LLM evaluation")
    p.add_argument(
        "--tasks", required=True, help=f"Comma-separated: {', '.join(TASKS.keys())}"
    )
    p.add_argument(
        "--model_args", required=True, help="model=...,base_url=...,num_concurrent=4"
    )
    p.add_argument(
        "--gen_kwargs", default="", help="e.g. temperature=0.7,max_tokens=1024"
    )
    p.add_argument("--max_samples", type=int, help="Max samples per task")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", help="Output JSON file")
    args = p.parse_args()

    m = _parse_kwargs(args.model_args)
    for key in ("model", "base_url"):
        if key not in m:
            p.error(f"model_args must include {key}=...")

    config = APIConfig(
        url=m["base_url"],
        model=m["model"],
        seed=args.seed,
        api_key=m.get("api_key", ""),
        num_concurrent=m.get("num_concurrent", 8),
        max_retries=m.get("max_retries", 3),
        gen_kwargs=_parse_kwargs(args.gen_kwargs),
    )
    output = asyncio.run(
        evaluate(
            [t.strip() for t in args.tasks.split(",") if t.strip()],
            config,
            args.max_samples,
        )
    )
    print(json.dumps(output, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
    return 0


if __name__ == "__main__":
    exit(main())
