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
import hashlib
import json
import logging
from pathlib import Path
from typing import TypedDict

from core import APIConfig, TaskResult, run_task
from tasks import TASKS


class ConfigInfo(TypedDict):
    model: str
    max_samples: int | None


class EvalResult(TypedDict):
    results: dict[str, TaskResult]
    dataset_hash: str
    total_seconds: float
    config: ConfigInfo


def _parse_kwargs(s: str) -> dict[str, str | int | float]:
    """Parse 'key=value,key=value' into dict."""
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid format '{pair}': expected 'key=value'")
        key, value = pair.split("=", 1)
        try:
            # Parse numbers/bools: temperature=0.7 -> 0.7, enabled=true -> True
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            # Unquoted strings fail JSON parsing, use as-is: model=gpt-4 -> "gpt-4"
            result[key] = value
    return result


def _write_samples_jsonl(path: Path, task_name: str, samples: list) -> None:
    """Write per-sample results to JSONL file."""
    filepath = path / f"samples_{task_name}.jsonl"
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


async def evaluate(
    task_names: list[str],
    config: APIConfig,
    max_samples: int | None = None,
    output_path: Path | None = None,
    log_samples: bool = False,
) -> EvalResult:
    """
    Run evaluations for specified tasks.

    Args:
        task_names: List of task names to evaluate
        config: API configuration
        max_samples: Optional limit on samples per task
        output_path: If provided, write results.json to this directory
        log_samples: If True, also write samples_{task}.jsonl files
    """
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, TaskResult] = {}
    task_hashes: list[str] = []
    total_seconds = 0.0

    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
        result = await run_task(TASKS[name], config, max_samples)
        task_hashes.append(result["task_hash"])
        if output_path and log_samples:
            _write_samples_jsonl(output_path, name, result["samples"])
        results[name] = {
            **result,
            "samples": [],
        }  # Exclude samples from main JSON output
        total_seconds += result["elapsed"]

    eval_result: EvalResult = {
        "results": results,
        "dataset_hash": hashlib.sha256(
            "".join(sorted(task_hashes)).encode()
        ).hexdigest(),
        "total_seconds": round(total_seconds, 2),
        "config": {"model": config.model, "max_samples": max_samples},
    }

    if output_path:
        with open(output_path / "results.json", "w") as f:
            json.dump(eval_result, f, indent=2)

    return eval_result


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="tinyeval - Minimal LLM evaluation")
    parser.add_argument(
        "--tasks", required=True, help=f"Comma-separated: {', '.join(TASKS.keys())}"
    )
    parser.add_argument(
        "--model_args",
        required=True,
        help="model=...,base_url=...,num_concurrent=4,max_retries=3",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help="Generation kwargs (e.g. temperature=0.7,max_tokens=1024)",
    )
    parser.add_argument("--max_samples", type=int, help="Max samples per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_path", help="Directory for results.json and sample files"
    )
    parser.add_argument(
        "--log_samples", action="store_true", help="Write per-sample JSONL files"
    )
    args = parser.parse_args()

    model_args = _parse_kwargs(args.model_args)

    if "model" not in model_args:
        parser.error("model_args must include model=...")
    if "base_url" not in model_args:
        parser.error("model_args must include base_url=...")

    base_url = str(model_args["base_url"]).rstrip("/")

    config = APIConfig(
        url=f"{base_url}/chat/completions",
        model=model_args["model"],
        seed=args.seed,
        api_key=model_args.get("api_key", ""),
        num_concurrent=model_args.get("num_concurrent", 8),
        max_retries=model_args.get("max_retries", 3),
        gen_kwargs=_parse_kwargs(args.gen_kwargs),
    )

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_path = Path(args.output_path) if args.output_path else None
    output = asyncio.run(
        evaluate(task_names, config, args.max_samples, output_path, args.log_samples)
    )

    print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
