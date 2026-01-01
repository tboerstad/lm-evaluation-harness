"""Core utilities for tinyeval - shared by main module and tasks."""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import aiohttp


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
    """Single request with retries. Returns response content or empty string."""
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
        except Exception as e:
            print(f"Request error (attempt {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)
    return ""


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
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            buf = BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        pass
    return image if isinstance(image, str) else ""


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r"[$,]", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    return text.lower().strip()
