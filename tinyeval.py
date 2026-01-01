#!/usr/bin/env python3
"""
tinyeval - Minimal LLM evaluation harness.

Two tasks: gsm8k_llama (text) and chartqa (multimodal).
One abstraction: batch HTTP requests to an OpenAI-compatible API.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import aiohttp
from PIL import Image

# Pre-compiled regex patterns for _normalize
_NORMALIZE_CURRENCY_RE = re.compile(r"[$,]")
_NORMALIZE_THOUGHT_RE = re.compile(r"(?s).*#### ")
_NORMALIZE_END_RE = re.compile(r"\.$")


@dataclass
class APIConfig:
    """API configuration."""

    url: str
    model: str
    api_key: str = ""
    num_concurrent: int = 8
    timeout: int = 300
    max_retries: int = 3
    seed: int = 1234


async def _request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> str:
    """Single request with retries. Raises RuntimeError if all retries fail."""
    for attempt in range(max_retries):
        try:
            async with semaphore, session.post(url, json=payload) as resp:
                if resp.ok:
                    data = await resp.json()
                    return (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                print(f"Request failed (attempt {attempt + 1}): {await resp.text()}")
        except asyncio.CancelledError:
            raise  # Allow the program to exit immediately on Ctrl+C
        except Exception as e:
            print(f"Request error (attempt {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)
    raise RuntimeError(
        f"Failed to get response from {url} after {max_retries} attempts"
    )


async def complete(
    prompts: list[str | tuple[str, list]],
    config: APIConfig,
    max_tokens: int = 512,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> list[str]:
    """
    Run batch of chat completions.

    Args:
        prompts: List of prompts. Each is either:
            - str: text-only prompt
            - tuple[str, list]: (text, images) for multimodal
        config: API configuration
        max_tokens: Max tokens per response
        temperature: Sampling temperature
        stop: Stop sequences (max 4)

    Returns:
        List of response strings
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    semaphore = asyncio.Semaphore(config.num_concurrent)
    connector = aiohttp.TCPConnector(limit=config.num_concurrent)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=config.timeout),
        headers=headers,
    ) as session:
        tasks = []
        for prompt in prompts:
            # Build messages
            if isinstance(prompt, tuple):
                text, images = prompt
                messages = _build_vision_message(text, images)
            else:
                messages = [{"role": "user", "content": prompt}]

            # Build payload
            payload: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": config.seed,
            }
            if stop:
                payload["stop"] = stop[:4]

            tasks.append(
                _request(session, config.url, payload, semaphore, config.max_retries)
            )

        return list(await asyncio.gather(*tasks))


def _build_vision_message(text: str, images: list) -> list[dict]:
    """Build OpenAI vision API message."""
    content: list[dict[str, Any]] = []
    for img in images:
        if b64 := _encode_image(img):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
    content.append({"type": "text", "text": text.replace("<image>", "").strip()})
    return [{"role": "user", "content": content}]


def _encode_image(image: Any) -> str:
    """Encode PIL image to base64, or pass through string."""
    if isinstance(image, str):
        assert not image.startswith("http"), "Remote image URLs are not supported."
        return image

    if isinstance(image, Image.Image):
        # Convert to RGB if needed to avoid save errors with CMYK/palette modes
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    return ""


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = _NORMALIZE_CURRENCY_RE.sub("", text)
    text = _NORMALIZE_THOUGHT_RE.sub("", text)
    text = _NORMALIZE_END_RE.sub("", text)
    return text.lower().strip()


async def run_task(
    task_name: str,
    config: APIConfig,
    docs: list,
    prompt_fn: Callable,
    max_tokens: int = 512,
    stop: list[str] | None = None,
) -> tuple[list[str], float]:
    """Shared execution loop: Format -> Timer -> Request -> Timer."""
    prompts = [prompt_fn(d) for d in docs]

    print(f"Evaluating: {task_name} ({len(docs)} samples)")
    t0 = time.perf_counter()
    responses = await complete(prompts, config, max_tokens=max_tokens, stop=stop)
    elapsed = time.perf_counter() - t0

    return responses, elapsed


async def evaluate(
    task_names: list[str], config: APIConfig, limit: int | None = None
) -> dict:
    """Run evaluations for specified tasks."""
    # Lazy import to avoid circular dependency: tasks import from tinyeval,
    # so we import TASKS here (at runtime) rather than at module level.
    from tasks import TASKS

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

    # Lazy import to avoid circular dependency (see evaluate() docstring)
    from tasks import TASKS

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
