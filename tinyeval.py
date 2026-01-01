#!/usr/bin/env python3
"""
tinyeval - Tiny Eval

A minimal, typed evaluation harness for LLMs via OpenAI-compatible APIs.

Supports two built-in tasks:
- gsm8k_llama: Text-based grade school math (chain-of-thought)
- chartqa: Multimodal chart question answering
"""
from __future__ import annotations

import asyncio
import base64
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import aiohttp
import datasets


# =============================================================================
# Task Imports
# =============================================================================

from tasks import TaskConfig, TASKS


@dataclass
class Instance:
    """Single evaluation instance."""

    doc: dict[str, Any]
    doc_id: int
    prompt: str
    target: str
    images: list[Any] = field(default_factory=list)
    response: str | None = None


@dataclass
class EvalResult:
    """Evaluation results."""

    task: str
    metrics: dict[str, float]
    num_samples: int


@dataclass
class APIConfig:
    """OpenAI-compatible API configuration."""

    base_url: str
    model: str
    api_key: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int = 1234
    num_concurrent: int = 8
    timeout: int = 300
    max_retries: int = 3


# =============================================================================
# Image Handling
# =============================================================================


def encode_image_to_base64(image: Any, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64."""
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            buf = BytesIO()
            image.save(buf, format=fmt)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        pass
    if isinstance(image, str):
        return image
    return ""


def get_images_from_doc(doc: dict[str, Any], image_fields: list[str]) -> list[Any]:
    """Extract images from document fields."""
    images = []
    for name in image_fields:
        if (img := doc.get(name)) is not None:
            images.extend(img if isinstance(img, list) else [img])
    return images


def build_multimodal_message(text: str, images: list[Any]) -> list[dict[str, Any]]:
    """Build vision API message with images."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        for img in images
        if (b64 := encode_image_to_base64(img))
    ]
    content.append({"type": "text", "text": text.replace("<image>", "").strip()})
    return [{"role": "user", "content": content}]


# =============================================================================
# Instance Building
# =============================================================================


def build_fewshot_context(config: TaskConfig) -> str:
    """Build few-shot context string from examples."""
    if not config.fewshot_examples:
        return ""
    parts = []
    for ex in config.fewshot_examples:
        prompt = config.doc_to_text(ex)
        target = ex.get("target", "")
        parts.append(f"{prompt} {target}")
    return "\n\n".join(parts) + "\n\n"


def build_instances(config: TaskConfig, docs: list[dict], limit: int | None = None) -> list[Instance]:
    """Convert documents to evaluation instances."""
    fewshot_context = build_fewshot_context(config)
    instances = []
    for doc_id, doc in enumerate(docs[:limit] if limit else docs):
        prompt = fewshot_context + config.doc_to_text(doc)
        target = config.doc_to_target(doc)
        images = get_images_from_doc(doc, config.doc_to_image) if config.doc_to_image else []
        instances.append(
            Instance(doc=doc, doc_id=doc_id, prompt=prompt, target=str(target), images=images)
        )
    return instances


# =============================================================================
# API Communication
# =============================================================================


def create_chat_payload(
    messages: list[dict[str, Any]],
    api_config: APIConfig,
    task_config: TaskConfig,
) -> dict[str, Any]:
    """Build chat completions request payload."""
    payload: dict[str, Any] = {
        "model": api_config.model,
        "messages": messages,
        "max_tokens": task_config.max_tokens,
        "temperature": task_config.temperature,
        "seed": api_config.seed,
    }
    if task_config.stop_sequences:
        payload["stop"] = task_config.stop_sequences[:4]  # OpenAI limit
    return payload


def create_text_message(text: str) -> list[dict[str, Any]]:
    """Create text-only user message."""
    return [{"role": "user", "content": text}]


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """POST request with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            async with semaphore, session.post(url, json=payload, headers=headers) as resp:
                if resp.ok:
                    return await resp.json()
                print(f"Request failed (attempt {attempt + 1}): {await resp.text()}")
        except Exception as e:
            print(f"Request error (attempt {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)
    return None


def parse_chat_response(response: dict[str, Any] | None) -> str:
    """Extract content from chat completion response."""
    if not response or not (choices := response.get("choices")):
        return ""
    return choices[0].get("message", {}).get("content", "")


async def run_generation(
    instances: list[Instance],
    api_config: APIConfig,
    task_config: TaskConfig,
) -> list[Instance]:
    """Run async generation requests for all instances."""
    if not instances:
        return instances

    headers = {"Content-Type": "application/json"}
    if api_config.api_key:
        headers["Authorization"] = f"Bearer {api_config.api_key}"

    semaphore = asyncio.Semaphore(api_config.num_concurrent)
    connector = aiohttp.TCPConnector(limit=api_config.num_concurrent)

    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=api_config.timeout)
    ) as session:
        tasks = []
        for inst in instances:
            if task_config.is_multimodal and inst.images:
                messages = build_multimodal_message(inst.prompt, inst.images)
            else:
                messages = create_text_message(inst.prompt)
            payload = create_chat_payload(messages, api_config, task_config)
            tasks.append(
                make_request(session, api_config.base_url, payload, headers, semaphore, api_config.max_retries)
            )
        for inst, response in zip(instances, await asyncio.gather(*tasks)):
            inst.response = parse_chat_response(response)

    return instances


# =============================================================================
# Metrics
# =============================================================================


def normalize_text(text: str, ignore_case: bool = True) -> str:
    """Normalize text for comparison."""
    text = re.sub(r",", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    if ignore_case:
        text = text.lower()
    return text.strip()


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 if normalized prediction matches reference."""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def relaxed_accuracy(prediction: str, reference: str) -> float:
    """ChartQA metric: exact match or 5% numeric tolerance."""

    def extract_answer(text: str) -> str:
        if match := re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE):
            return match.group(1).strip()
        return text.strip()

    def parse_number(s: str) -> float | None:
        try:
            return float(s.replace(",", "").replace("%", "").replace("$", "").strip())
        except ValueError:
            return None

    pred = extract_answer(prediction)
    ref = str(reference).strip()

    if pred.lower() == ref.lower():
        return 1.0

    pred_num, ref_num = parse_number(pred), parse_number(ref)
    if pred_num is not None and ref_num is not None:
        if ref_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        if abs(pred_num - ref_num) / abs(ref_num) <= 0.05:
            return 1.0

    return 0.0


def compute_metrics(instances: list[Instance], config: TaskConfig) -> dict[str, float]:
    """Compute metrics over all instances."""
    exact_scores = []
    relaxed_scores = []

    for inst in instances:
        response = inst.response or ""

        # Apply answer extraction if defined
        if config.extract_answer:
            response = config.extract_answer(response)

        exact_scores.append(exact_match(response, inst.target))
        relaxed_scores.append(relaxed_accuracy(response, inst.target))

    n = len(instances)
    return {
        "exact_match": sum(exact_scores) / n if n else 0.0,
        "relaxed_accuracy": sum(relaxed_scores) / n if n else 0.0,
    }


# =============================================================================
# Dataset Loading
# =============================================================================


def load_docs(dataset_path: str, dataset_name: str | None, limit: int | None = None) -> list[dict]:
    """Load documents from HuggingFace dataset."""
    if limit:
        for split in ["test", "validation", "train"]:
            try:
                ds = datasets.load_dataset(dataset_path, dataset_name, split=split, streaming=True)
                return list(ds.take(limit))
            except ValueError:
                continue
        raise ValueError(f"No valid split found in {dataset_path}")

    ds = datasets.load_dataset(path=dataset_path, name=dataset_name)
    return [doc for split in ds for doc in ds[split]]


# =============================================================================
# Evaluation
# =============================================================================


async def evaluate_task(
    task_name: str,
    api_config: APIConfig,
    limit: int | None = None,
) -> tuple[EvalResult, float]:
    """Evaluate a single task."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    config = TASKS[task_name]
    print(f"Evaluating: {config.task} (multimodal: {config.is_multimodal})")

    docs = load_docs(config.dataset_path, config.dataset_name, limit)
    instances = build_instances(config, docs, limit)
    print(f"Built {len(instances)} instances")

    start_time = time.perf_counter()
    instances = await run_generation(instances, api_config, config)
    eval_time = time.perf_counter() - start_time

    metrics = compute_metrics(instances, config)
    return EvalResult(task=config.task, metrics=metrics, num_samples=len(instances)), eval_time


async def evaluate_tasks(
    task_names: list[str],
    api_config: APIConfig,
    limit: int | None = None,
) -> tuple[dict[str, tuple[EvalResult, float]], float]:
    """Evaluate multiple tasks sequentially."""
    results = {}
    total_time = 0.0
    for name in task_names:
        result, eval_time = await evaluate_task(name, api_config, limit)
        results[result.task] = (result, eval_time)
        total_time += eval_time
        print(f"{result.task}: {result.metrics} ({eval_time:.2f}s)")
    return results, total_time


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """CLI entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="tinyeval - Tiny Eval")
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help=f"Comma-separated task names. Available: {', '.join(TASKS.keys())}",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base_url", type=str, required=True, help="API base URL")
    parser.add_argument("--api_key", type=str, default="", help="API key")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--num_concurrent", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",")]

    api_config = APIConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        num_concurrent=args.num_concurrent,
        seed=args.seed,
    )

    results, total_time = asyncio.run(evaluate_tasks(task_names, api_config, args.limit))

    output = {
        "results": {
            name: {"metrics": r.metrics, "num_samples": r.num_samples, "time_seconds": round(t, 2)}
            for name, (r, t) in results.items()
        },
        "config": {"model": args.model, "limit": args.limit},
        "total_time_seconds": round(total_time, 2),
    }

    print(json.dumps(output, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
