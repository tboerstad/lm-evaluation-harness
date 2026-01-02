"""Core utilities for tinyeval."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, TypedDict

import datasets
import httpx
from PIL import Image

logger = logging.getLogger(__name__)


def load_samples(
    dataset: str,
    splits: list[str],
    make_sample: Callable[[dict], Sample],
    max_samples: int | None = None,
    config: str | None = None,
) -> list[Sample]:
    """Load samples from HuggingFace dataset across splits."""
    result: list[Sample] = []
    remaining = max_samples
    for split in splits:
        if remaining is not None and remaining <= 0:
            break
        ds = datasets.load_dataset(dataset, config, split=split, streaming=True)
        for doc in ds.take(remaining) if remaining is not None else ds:
            result.append(make_sample(doc))
        if remaining is not None:
            remaining = max_samples - len(result)
    return result


class Metrics(TypedDict):
    exact_match: float


class TaskResult(TypedDict):
    task: str
    metrics: Metrics
    num_samples: int
    elapsed: float


@dataclass
class Sample:
    """A single evaluation sample: prompt + expected target."""

    prompt: str | tuple[str, list[Any]]
    target: str


@dataclass(frozen=True)
class Task:
    """Minimal task definition: a loader of samples + a scoring function."""

    name: str
    samples: Callable[[int | None], list[Sample]]
    score: Callable[[str, str], float]


MAX_BACKOFF = 8  # Cap exponential backoff at 8 seconds


@dataclass
class APIConfig:
    """API configuration."""

    url: str
    model: str
    seed: int
    api_key: str = ""
    num_concurrent: int = 8
    timeout: int = 300
    max_retries: int = 3
    gen_kwargs: dict[str, Any] = field(default_factory=dict)


async def _request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    max_retries: int,
) -> str:
    """Single request with retries. Raises RuntimeError if all retries fail."""
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload)
            if resp.is_success:
                return resp.json()["choices"][0]["message"]["content"]
            logger.warning("Request failed (attempt %d): %s", attempt + 1, resp.text)
        except asyncio.CancelledError:
            raise  # Allow the program to exit immediately on Ctrl+C
        except httpx.HTTPError as e:
            logger.warning("Request error (attempt %d): %s", attempt + 1, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(min(2**attempt, MAX_BACKOFF))
    raise RuntimeError(
        f"Failed to get response from {url} after {max_retries} attempts"
    )


async def complete(
    prompts: list[str | tuple[str, list]], config: APIConfig
) -> list[str]:
    """Run batch of chat completions. Prompts can be str or (text, images) tuples."""
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=config.num_concurrent),
        timeout=httpx.Timeout(config.timeout),
        headers=headers,
        trust_env=True,
    ) as client:
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
                **config.gen_kwargs,
            }

            tasks.append(_request(client, config.url, payload, config.max_retries))

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
        if image.startswith("http"):
            raise ValueError("Remote image URLs are not supported.")
        return image

    if isinstance(image, Image.Image):
        # Convert to RGB if needed to avoid save errors with CMYK/palette modes
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r"(?s).*#### ", "", text)
    return re.sub(r"[$,]|\.$", "", text).lower().strip()


async def run_task(
    task: Task, config: APIConfig, max_samples: int | None = None
) -> TaskResult:
    """Evaluate a task: collect samples, run inference, compute scores."""
    samples = task.samples(max_samples)
    prompts = [s.prompt for s in samples]

    logger.info("Evaluating: %s (%d samples)", task.name, len(samples))
    t0 = time.perf_counter()
    responses = await complete(prompts, config)
    elapsed = time.perf_counter() - t0

    # Score each response
    scores = [task.score(r, s.target) for r, s in zip(responses, samples)]
    accuracy = sum(scores) / len(samples) if samples else 0.0

    logger.info("%s: accuracy=%.4f (%.2fs)", task.name, accuracy, elapsed)

    return {
        "task": task.name,
        "metrics": {"exact_match": accuracy},
        "num_samples": len(samples),
        "elapsed": round(elapsed, 2),
    }
