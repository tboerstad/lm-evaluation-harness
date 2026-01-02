"""
Core utilities for tinyeval.

Responsibilities:
- APIConfig: endpoint, model, concurrency, timeout
- Sample/Task: minimal task abstraction (generator + scorer)
- complete(): async batch chat completions (OpenAI-compatible)
- run_task(): evaluate a Task, return TaskResult
- _normalize(): text normalization for comparison
- _encode_image(): PIL→base64; rejects remote URLs
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import BytesIO
from typing import Any, TypedDict

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)


class Metrics(TypedDict):
    exact_match: float
    relaxed_accuracy: float


class TaskResult(TypedDict):
    task: str
    metrics: Metrics
    num_samples: int
    elapsed: float


@dataclass
class Sample:
    """A single evaluation sample: prompt + expected target."""

    prompt: str | tuple[str, list]  # text or (text, images) for multimodal
    target: str


@dataclass
class Task:
    """
    Minimal task definition: a generator of samples + a scoring function.

    Examples:
        # Text-only task
        Task(
            name="gsm8k",
            samples=lambda n: (Sample(prompt, target) for prompt, target in data[:n]),
            score=lambda response, target: 1.0 if response == target else 0.0,
        )

        # Multimodal task
        Task(
            name="chartqa",
            samples=lambda n: (Sample((text, [img]), target) for ...),
            score=relaxed_match,
        )
    """

    name: str
    samples: Callable[[int | None], Iterator[Sample]]  # max_samples -> samples
    score: Callable[[str, str], float]  # (response, target) -> score


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
    payload: dict[str, Any],
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
                logger.warning(
                    "Request failed (attempt %d): %s", attempt + 1, await resp.text()
                )
        except asyncio.CancelledError:
            raise  # Allow the program to exit immediately on Ctrl+C
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.warning("Request error (attempt %d): %s", attempt + 1, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)
    raise RuntimeError(
        f"Failed to get response from {url} after {max_retries} attempts"
    )


async def complete(
    prompts: list[str | tuple[str, list]],
    config: APIConfig,
    gen_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """
    Run batch of chat completions.

    Args:
        prompts: List of prompts. Each is either:
            - str: text-only prompt
            - tuple[str, list]: (text, images) for multimodal
        config: API configuration
        gen_kwargs: Generation kwargs forwarded to API (e.g. max_tokens, temperature, stop)

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
            if isinstance(prompt, tuple):
                text, images = prompt
                messages = _build_vision_message(text, images)
            else:
                messages = [{"role": "user", "content": prompt}]

            payload: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "seed": config.seed,
            }
            if gen_kwargs:
                payload.update(gen_kwargs)

            tasks.append(
                _request(session, config.url, payload, semaphore, config.max_retries)
            )

        return list(await asyncio.gather(*tasks))


def _build_vision_message(text: str, images: list[Any]) -> list[dict[str, Any]]:
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
    task: Task,
    config: APIConfig,
    max_samples: int | None = None,
    gen_kwargs: dict[str, Any] | None = None,
) -> TaskResult:
    """
    Evaluate a task: collect samples, run inference, compute scores.

    Args:
        task: Task definition with samples generator and scoring function
        config: API configuration
        max_samples: Optional limit on number of samples
        gen_kwargs: Generation kwargs forwarded to API (e.g. max_tokens, temperature)

    Returns:
        TaskResult with metrics, sample count, and elapsed time
    """
    samples = list(task.samples(max_samples))
    prompts = [s.prompt for s in samples]

    logger.info("Evaluating: %s (%d samples)", task.name, len(samples))
    t0 = time.perf_counter()
    responses = await complete(prompts, config, gen_kwargs)
    elapsed = time.perf_counter() - t0

    # Score each response
    scores = [task.score(r, s.target) for r, s in zip(responses, samples)]
    accuracy = sum(scores) / len(samples) if samples else 0.0

    logger.info("%s: accuracy=%.4f (%.2fs)", task.name, accuracy, elapsed)

    return {
        "task": task.name,
        "metrics": {"exact_match": accuracy, "relaxed_accuracy": accuracy},
        "num_samples": len(samples),
        "elapsed": round(elapsed, 2),
    }
