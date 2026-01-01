#!/usr/bin/env python3
"""
tinyeval - Minimal LLM evaluation harness.
Combines API client, task registry, and CLI in one file.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import aiohttp
import datasets
from PIL import Image

# ==========================================
# 1. Core Engine & API Client
# ==========================================


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
            raise
        except Exception as e:
            print(f"Request error (attempt {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)
    raise RuntimeError(
        f"Failed to get response from {url} after {max_retries} attempts"
    )


def _encode_image(image: Any) -> str:
    """Encode PIL image to base64, or pass through string."""
    if isinstance(image, str):
        assert not image.startswith("http"), "Remote image URLs are not supported."
        return image
    if isinstance(image, Image.Image):
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    return ""


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
            if isinstance(prompt, tuple):
                text, images = prompt
                messages = _build_vision_message(text, images)
            else:
                messages = [{"role": "user", "content": prompt}]

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


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r"[$,]", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    return text.lower().strip()


# ==========================================
# 2. Task Definitions (GSM8K & ChartQA)
# ==========================================

# --- GSM8K Data & Logic ---

GSM8K_FEWSHOT = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8",
    ),
]

GSM8K_STOP = [
    "<|eot_id|>",
    "<|start_header_id|>user<|end_header_id|>",
    "Q:",
    "</s>",
    "<|im_end|>",
]

_GSM8K_TEMPLATE = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
)

_NUM_PATTERN = r"-?[$0-9.,]{2,}|-?[0-9]+"


def _format_gsm8k_prompt(question: str) -> str:
    """Format GSM8K prompt with few-shot examples."""
    parts = [_GSM8K_TEMPLATE.format(question=q) + f"\n {a}" for q, a in GSM8K_FEWSHOT]
    parts.append(_GSM8K_TEMPLATE.format(question=question))
    return "\n\n".join(parts)


def _parse_gsm8k_target(answer: str) -> str:
    """Parse target answer from GSM8K format, handling missing #### delimiter."""
    parts = answer.split("####")
    if len(parts) < 2:
        return answer.strip()
    return parts[-1].strip()


def _extract_gsm8k_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response."""
    if match := re.search(rf"The final answer is ({_NUM_PATTERN})", response):
        return match.group(1)
    matches = re.findall(rf"({_NUM_PATTERN})", response)
    if matches:
        return matches[-1]
    return response


# --- ChartQA Data & Logic ---

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_NUMERIC_CLEAN_RE = re.compile(r"[$,%]")


def _format_chartqa_prompt(query: str) -> str:
    """Format ChartQA prompt."""
    return (
        f"<image>You are provided a chart image and will be asked a question. "
        f"You have to think through your answer and provide a step-by-step solution. "
        f'Once you have the solution, write the final answer in at most a few words at the end with the phrase "FINAL ANSWER:". '
        f"The question is: {query}\n"
        f"Let's think step by step."
    )


def _relaxed_match(response: str, target: str) -> float:
    """ChartQA metric: exact match or 5% numeric tolerance."""
    if match := _FINAL_ANSWER_RE.search(response):
        pred = match.group(1).strip()
    else:
        pred = response.strip()

    if pred.lower() == target.lower():
        return 1.0

    try:
        pred_n = float(_NUMERIC_CLEAN_RE.sub("", pred))
        target_n = float(_NUMERIC_CLEAN_RE.sub("", target))
        if target_n == 0:
            return 1.0 if pred_n == 0 else 0.0
        if abs(pred_n - target_n) / abs(target_n) <= 0.05:
            return 1.0
    except ValueError:
        pass

    return 0.0


# ==========================================
# 3. Registry & Execution Loop
# ==========================================


@dataclass
class TaskDef:
    """Task definition with loader, prompt, and scoring functions."""

    loader: Callable[[int | None], list]
    prompt_fn: Callable
    score_fn: Callable[[str, str], tuple[bool, float]]
    target_fn: Callable[[dict], str]
    stop: list[str] | None = None


def _load_gsm8k(limit: int | None) -> list:
    """Load GSM8K test set."""
    ds = datasets.load_dataset("gsm8k", "main", split="test", streaming=True)
    return list(ds.take(limit) if limit else ds)


def _load_chartqa(limit: int | None) -> list:
    """Load ChartQA dataset, trying test/validation/train splits."""
    for split in ["test", "validation", "train"]:
        try:
            ds = datasets.load_dataset(
                "HuggingFaceM4/ChartQA", split=split, streaming=True
            )
            return list(ds.take(limit) if limit else ds)
        except ValueError:
            continue
    raise ValueError("No valid split found in HuggingFaceM4/ChartQA")


def _score_gsm8k(response: str, target: str) -> tuple[bool, float]:
    """Score GSM8K response. Returns (exact_match, relaxed_accuracy)."""
    pred = _extract_gsm8k_answer(response)
    is_correct = _normalize(pred) == _normalize(target)
    return is_correct, float(is_correct)


def _score_chartqa(response: str, target: str) -> tuple[bool, float]:
    """Score ChartQA response. Returns (exact_match, relaxed_accuracy)."""
    if match := _FINAL_ANSWER_RE.search(response):
        pred = match.group(1).strip()
    else:
        pred = response.strip()
    exact = _normalize(pred) == _normalize(target)
    relaxed = _relaxed_match(response, target)
    return exact, relaxed


TASKS: dict[str, TaskDef] = {
    "gsm8k_llama": TaskDef(
        loader=_load_gsm8k,
        prompt_fn=lambda d: _format_gsm8k_prompt(d["question"]),
        score_fn=_score_gsm8k,
        target_fn=lambda d: _parse_gsm8k_target(d["answer"]),
        stop=GSM8K_STOP,
    ),
    "chartqa": TaskDef(
        loader=_load_chartqa,
        prompt_fn=lambda d: (_format_chartqa_prompt(d["query"]), [d["image"]]),
        score_fn=_score_chartqa,
        target_fn=lambda d: d["label"][0]
        if isinstance(d["label"], list)
        else str(d["label"]),
    ),
}


async def run_eval(task_name: str, config: APIConfig, limit: int | None) -> dict:
    """Unified evaluation loop."""
    task = TASKS[task_name]
    docs = task.loader(limit)
    targets = [task.target_fn(d) for d in docs]

    print(f"Evaluating: {task_name} ({len(docs)} samples)")

    prompts = [task.prompt_fn(d) for d in docs]
    t0 = time.perf_counter()
    responses = await complete(prompts, config, stop=task.stop)
    elapsed = time.perf_counter() - t0

    scores = [task.score_fn(r, t) for r, t in zip(responses, targets)]

    metrics = {
        "exact_match": sum(s[0] for s in scores) / len(docs),
        "relaxed_accuracy": sum(s[1] for s in scores) / len(docs),
    }
    print(f"{task_name}: {metrics} ({elapsed:.2f}s)")

    return {
        "task": task_name,
        "metrics": metrics,
        "num_samples": len(docs),
        "time_seconds": round(elapsed, 2),
    }


# ==========================================
# 4. CLI Entry Point
# ==========================================


async def evaluate(
    task_names: list[str], config: APIConfig, limit: int | None = None
) -> dict:
    """Run evaluations for specified tasks."""
    results = {}
    total_time = 0.0

    for name in task_names:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
        result = await run_eval(name, config, limit)
        results[name] = result
        total_time += result["time_seconds"]

    return {"results": results, "total_time_seconds": round(total_time, 2)}


def main() -> int:
    """CLI entry point."""
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
