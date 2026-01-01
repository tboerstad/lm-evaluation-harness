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
# Task Definitions (Python, no YAML)
# =============================================================================

# GSM8K Llama - Chain of Thought few-shot examples
GSM8K_FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8",
    },
]


def gsm8k_doc_to_text(doc: dict[str, Any]) -> str:
    """Convert GSM8K document to prompt text."""
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {doc['question']}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    )


def gsm8k_doc_to_target(doc: dict[str, Any]) -> str:
    """Extract target answer from GSM8K document."""
    answer = doc.get("answer", doc.get("target", ""))
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer


def gsm8k_extract_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response using regex filters."""
    # Try strict pattern first
    match = re.search(r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))", response)
    if match:
        return match.groups()[-1] or match.group(1)
    # Fall back to flexible pattern
    match = re.search(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    if match:
        return match.groups()[-1] or match.group(1)
    return response


def chartqa_doc_to_text(doc: dict[str, Any]) -> str:
    """Convert ChartQA document to prompt text."""
    return (
        f"<image>You are provided a chart image and will be asked a question. "
        f"You have to think through your answer and provide a step-by-step solution. "
        f'Once you have the solution, write the final answer in at most a few words at the end with the phrase "FINAL ANSWER:". '
        f"The question is: {doc['query']}\n"
        f"Let's think step by step."
    )


def chartqa_doc_to_target(doc: dict[str, Any]) -> str:
    """Extract target answer from ChartQA document."""
    label = doc.get("label", [])
    if isinstance(label, list) and label:
        return label[0]
    return str(label)


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class TaskConfig:
    """Task configuration for evaluation."""

    task: str
    dataset_path: str
    dataset_name: str | None = None
    doc_to_text: Any = None  # Callable[[dict], str]
    doc_to_target: Any = None  # Callable[[dict], str]
    extract_answer: Any = None  # Optional callable to extract answer from response
    doc_to_image: list[str] | None = None  # Field names containing images
    fewshot_examples: list[dict] | None = None
    stop_sequences: list[str] = field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.0

    @property
    def is_multimodal(self) -> bool:
        return self.doc_to_image is not None


# Built-in task configs
TASKS: dict[str, TaskConfig] = {
    "gsm8k_llama": TaskConfig(
        task="gsm8k_llama",
        dataset_path="gsm8k",
        dataset_name="main",
        doc_to_text=gsm8k_doc_to_text,
        doc_to_target=gsm8k_doc_to_target,
        extract_answer=gsm8k_extract_answer,
        fewshot_examples=GSM8K_FEWSHOT_EXAMPLES,
        stop_sequences=["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "Q:", "</s>", "<|im_end|>"],
        max_tokens=512,
    ),
    "chartqa": TaskConfig(
        task="chartqa",
        dataset_path="HuggingFaceM4/ChartQA",
        doc_to_text=chartqa_doc_to_text,
        doc_to_target=chartqa_doc_to_target,
        doc_to_image=["image"],
        max_tokens=512,
    ),
}


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
                messages = [{"role": "user", "content": inst.prompt}]
            payload = create_chat_payload(messages, api_config, task_config)
            tasks.append(
                make_request(session, api_config.base_url, payload, headers, semaphore, api_config.max_retries)
            )
        for inst, resp in zip(instances, await asyncio.gather(*tasks)):
            inst.response = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ""

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
) -> tuple[dict[str, Any], float]:
    """Evaluate a single task. Returns (result_dict, eval_time)."""
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
    return {"task": config.task, "metrics": metrics, "num_samples": len(instances)}, eval_time


async def evaluate_tasks(
    task_names: list[str],
    api_config: APIConfig,
    limit: int | None = None,
) -> tuple[dict[str, tuple[dict[str, Any], float]], float]:
    """Evaluate multiple tasks sequentially."""
    results = {}
    total_time = 0.0
    for name in task_names:
        result, eval_time = await evaluate_task(name, api_config, limit)
        results[result["task"]] = (result, eval_time)
        total_time += eval_time
        print(f"{result['task']}: {result['metrics']} ({eval_time:.2f}s)")
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
            name: {"metrics": r["metrics"], "num_samples": r["num_samples"], "time_seconds": round(t, 2)}
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
