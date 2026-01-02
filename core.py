"""
Core utilities for tinyeval.

Responsibilities:
- APIConfig: endpoint, model, concurrency, timeout
- complete(): async batch chat completions (OpenAI-compatible)
- run_task(): format prompts, time requests, return responses
- _normalize(): text normalization for comparison
- _encode_image(): PIL→base64; rejects remote URLs
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import TypedDict

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)

# Type alias for image inputs (PIL Image or base64 string)
ImageInput = Image.Image | str


class ImageURLDetail(TypedDict):
    url: str


class ImageContentPart(TypedDict):
    type: str
    image_url: ImageURLDetail


class TextContentPart(TypedDict):
    type: str
    text: str


ContentPart = ImageContentPart | TextContentPart


class ChatMessage(TypedDict):
    role: str
    content: str | list[ContentPart]


# Document type: row from a dataset with string keys
Document = dict[str, object]


class Metrics(TypedDict):
    exact_match: float
    relaxed_accuracy: float


class TaskResult(TypedDict):
    task: str
    metrics: Metrics
    num_samples: int
    elapsed: float


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
    payload: dict[str, str | int | float | list[str] | list[ChatMessage]],
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
    prompts: list[str | tuple[str, list[ImageInput]]],
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
            if isinstance(prompt, tuple):
                text, images = prompt
                messages = _build_vision_message(text, images)
            else:
                messages = [{"role": "user", "content": prompt}]

            payload: dict[str, str | int | float | list[str] | list[ChatMessage]] = {
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


def _build_vision_message(text: str, images: list[ImageInput]) -> list[ChatMessage]:
    """Build OpenAI vision API message."""
    content: list[ContentPart] = []
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


def _encode_image(image: ImageInput) -> str:
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
    docs: list[Document],
    prompt_fn: Callable[[Document], str | tuple[str, list[ImageInput]]],
    max_tokens: int = 512,
    stop: list[str] | None = None,
) -> tuple[list[str], float]:
    """Shared execution loop: Format -> Timer -> Request -> Timer."""
    prompts = [prompt_fn(d) for d in docs]

    logger.info("Evaluating: %s (%d samples)", task_name, len(docs))
    t0 = time.perf_counter()
    responses = await complete(prompts, config, max_tokens=max_tokens, stop=stop)
    elapsed = time.perf_counter() - t0

    return responses, elapsed
