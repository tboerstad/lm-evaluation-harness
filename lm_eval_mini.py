#!/usr/bin/env python3
"""
Minimal LM Evaluation Harness - A clean, typed implementation.

Supports:
- YAML task configs with jinja2 templates
- Async HTTP requests to OpenAI-compatible APIs (chat completions)
- Text generation tasks (GSM8K, etc.)
- Multimodal tasks with images (ChartQA)
- Core metrics: exact_match, relaxed_accuracy
"""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import random
import re
import string
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
import datasets
import jinja2
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# Types
# ============================================================================


@dataclass
class TaskConfig:
    """YAML task configuration (dataset, prompts, metrics, generation settings).

    Note: This mini implementation loads all available splits combined. This is
    intentional - the mini harness is for relative comparisons between inference
    frameworks, not for train/val separation.
    """
    task: str  # Unique identifier for the task (e.g., "gsm8k", "chartqa")
    dataset_path: str  # HuggingFace dataset path or local directory
    dataset_name: str | None = None  # Dataset configuration/subset name (if applicable)
    num_fewshot: int = 0  # Number of few-shot examples to prepend to each prompt
    doc_to_text: str | None = None  # Jinja2 template to convert document to input prompt
    doc_to_target: str | None = None  # Jinja2 template or field name for the expected answer
    doc_to_image: list[str] | str | None = None  # Document field(s) containing images for multimodal tasks
    target_delimiter: str = " "  # Separator between prompt and target in few-shot examples
    fewshot_delimiter: str = "\n\n"  # Separator between few-shot examples
    generation_kwargs: dict[str, Any] = field(default_factory=dict)  # API generation params (temperature, max_tokens, until)
    metric_list: list[dict[str, Any]] = field(default_factory=list)  # Metrics to compute (exact_match, relaxed_accuracy, etc.)
    filter_list: list[dict[str, Any]] | None = None  # Post-processing filters to extract answers from responses

    @classmethod
    def from_yaml(cls, path: Path) -> TaskConfig:
        """Load config from YAML, handling !function tags and includes."""
        class FunctionTagLoader(yaml.SafeLoader):
            pass
        FunctionTagLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))

        with open(path) as f:
            data = yaml.load(f, Loader=FunctionTagLoader)

        if "include" in data:
            base = cls.from_yaml(path.parent / data.pop("include"))
            for key, value in data.items():
                setattr(base, key, value)
            return base

        data.setdefault("task", path.stem)
        if "doc_to_image" in data and isinstance(data["doc_to_image"], str):
            data["doc_to_image"] = [data["doc_to_image"]]

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_multimodal(self) -> bool:
        return self.doc_to_image is not None


@dataclass
class Instance:
    """Single evaluation instance: prompt, target, optional images, and response."""
    doc: dict[str, Any]  # Original document from the dataset
    doc_id: int  # Index of this document in the dataset
    prompt: str  # Rendered prompt text (including few-shot context if any)
    target: str | list[str]  # Expected answer(s) for evaluation
    images: list[Any] = field(default_factory=list)  # PIL Images or base64 strings for multimodal tasks
    generation_kwargs: dict[str, Any] = field(default_factory=dict)  # Per-instance generation parameters
    response: str | None = None  # Model-generated response (populated after API call)


@dataclass
class EvalResult:
    """Evaluation results: task name, computed metrics, sample count."""
    task: str  # Name of the evaluated task
    metrics: dict[str, float]  # Computed metric scores (e.g., {"exact_match": 0.85})
    num_samples: int  # Total number of evaluated samples


# ============================================================================
# Template Engine
# ============================================================================

_jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)


def render_template(template: str, doc: dict[str, Any]) -> str:
    """Render jinja2 template with document as context."""
    try:
        return _jinja_env.from_string(template).render(**doc)
    except jinja2.UndefinedError as e:
        log.warning(f"Template error: {e}")
        return template


def resolve_field(doc: dict[str, Any], field_spec: str | list | int | None) -> Any:
    """Resolve field spec: return literal, doc field, or rendered template."""
    if field_spec is None or isinstance(field_spec, (int, list)):
        return field_spec
    if field_spec in doc:
        return doc[field_spec]
    result = render_template(field_spec, doc)
    return int(result) if result.isdigit() else result


# ============================================================================
# Image Handling
# ============================================================================

def encode_image_to_base64(image: Any, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64. Strings (URLs/base64) pass through unchanged."""
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
    log.warning(f"Unknown image type: {type(image)}")
    return ""


def get_images_from_doc(doc: dict[str, Any], image_fields: list[str]) -> list[Any]:
    """Extract images from doc fields, flattening lists."""
    images = []
    for name in image_fields:
        if (img := doc.get(name)) is not None:
            images.extend(img if isinstance(img, list) else [img])
    return images


def build_multimodal_message(text: str, images: list[Any]) -> list[dict[str, Any]]:
    """Build vision API message: images first, then text (with <image> removed)."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        for img in images if (b64 := encode_image_to_base64(img))
    ]
    content.append({"type": "text", "text": text.replace("<image>", "").strip()})
    return [{"role": "user", "content": content}]


# ============================================================================
# Task Loading and Instance Building
# ============================================================================

def get_fewshot_examples(
    docs: list[dict], num_fewshot: int, rng: random.Random, exclude_indices: set[int] | None = None,
) -> list[dict]:
    """Sample num_fewshot examples from docs, optionally excluding certain indices."""
    if num_fewshot == 0:
        return []
    exclude_indices = exclude_indices or set()
    available = [(i, doc) for i, doc in enumerate(docs) if i not in exclude_indices]
    if not available:
        return []
    sampled = rng.sample(available, min(num_fewshot, len(available)))
    return [doc for _, doc in sampled]


def build_fewshot_context(config: TaskConfig, examples: list[dict]) -> str:
    """Format few-shot examples as context string."""
    if not examples:
        return ""
    parts = [
        f"{resolve_field(doc, config.doc_to_text)}{config.target_delimiter}{resolve_field(doc, config.doc_to_target)}"
        for doc in examples
    ]
    return config.fewshot_delimiter.join(parts) + config.fewshot_delimiter


def build_instances(
    config: TaskConfig, docs: list[dict], fewshot_context: str = "", limit: int | None = None,
) -> list[Instance]:
    """Convert documents to evaluation instances with prompts and targets."""
    instances = []
    for doc_id, doc in enumerate(docs[:limit] if limit else docs):
        target = resolve_field(doc, config.doc_to_target)
        instances.append(Instance(
            doc=doc,
            doc_id=doc_id,
            prompt=fewshot_context + str(resolve_field(doc, config.doc_to_text)),
            target=target if isinstance(target, list) else str(target),
            images=get_images_from_doc(doc, config.doc_to_image) if config.doc_to_image else [],
            generation_kwargs=config.generation_kwargs.copy(),
        ))
    return instances


# ============================================================================
# HTTP Client for OpenAI-compatible APIs
# ============================================================================

@dataclass
class APIConfig:
    """OpenAI-compatible API configuration."""
    base_url: str  # Full URL to the chat completions endpoint (e.g., "http://localhost:8000/v1/chat/completions")
    model: str  # Model name to pass in API requests
    api_key: str = ""  # Bearer token for Authorization header (empty for local APIs)
    max_tokens: int = 512  # Default maximum tokens to generate per request
    temperature: float = 0.0  # Sampling temperature (0.0 = deterministic)
    seed: int = 1234  # Random seed for reproducible generation
    num_concurrent: int = 8  # Maximum number of concurrent HTTP requests
    timeout: int = 300  # Request timeout in seconds
    max_retries: int = 3  # Number of retry attempts on request failure


def handle_stop_sequences(until: list[str] | str | None) -> list[str]:
    """Normalize stop sequences to list, filtering empties."""
    if until is None:
        return []
    if isinstance(until, str):
        return [until] if until else []
    return [s for s in until if s]


def create_chat_payload(
    messages: list[dict[str, Any]], config: APIConfig, gen_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build chat completions request payload."""
    gen_kwargs = gen_kwargs or {}
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": gen_kwargs.get("max_tokens") or gen_kwargs.get("max_gen_toks", config.max_tokens),
        "temperature": gen_kwargs.get("temperature", config.temperature),
        "seed": config.seed,
    }
    if stop := handle_stop_sequences(gen_kwargs.get("until")):
        payload["stop"] = stop[:4]  # OpenAI limits to 4
    return payload


def create_text_message(text: str) -> list[dict[str, Any]]:
    """Create simple text-only user message."""
    return [{"role": "user", "content": text}]


async def make_request(
    session: aiohttp.ClientSession, url: str, payload: dict[str, Any],
    headers: dict[str, str], semaphore: asyncio.Semaphore, max_retries: int = 3,
) -> dict[str, Any] | None:
    """POST request with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            async with semaphore, session.post(url, json=payload, headers=headers) as resp:
                if resp.ok:
                    return await resp.json()
                log.warning(f"Request failed (attempt {attempt + 1}): {await resp.text()}")
        except Exception as e:
            log.warning(f"Request error (attempt {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    return None


def parse_chat_response(response: dict[str, Any] | None) -> str:
    """Extract content from chat completion response."""
    if not response or not (choices := response.get("choices")):
        return ""
    return choices[0].get("message", {}).get("content", "")


async def run_generation(
    instances: list[Instance], config: APIConfig, is_multimodal: bool = False,
) -> list[Instance]:
    """Run async generation requests for all instances, populating responses."""
    if not instances:
        return instances

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    semaphore = asyncio.Semaphore(config.num_concurrent)
    connector = aiohttp.TCPConnector(limit=config.num_concurrent)

    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
        tasks = [
            make_request(
                session, config.base_url,
                create_chat_payload(
                    build_multimodal_message(inst.prompt, inst.images) if is_multimodal and inst.images
                    else create_text_message(inst.prompt),
                    config, inst.generation_kwargs
                ),
                headers, semaphore, config.max_retries
            )
            for inst in instances
        ]
        for inst, response in zip(instances, await asyncio.gather(*tasks)):
            inst.response = parse_chat_response(response)

    return instances


# ============================================================================
# Metrics
# ============================================================================

def normalize_text(
    text: str, ignore_case: bool = False, ignore_punctuation: bool = False,
    regexes_to_ignore: list[str] | None = None,
) -> str:
    """Normalize text: apply regex removals, case folding, punctuation removal."""
    for pattern in (regexes_to_ignore or []):
        text = re.sub(pattern, "", text)
    if ignore_case:
        text = text.lower()
    if ignore_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def exact_match(
    prediction: str, reference: str | list[str], ignore_case: bool = False,
    ignore_punctuation: bool = False, regexes_to_ignore: list[str] | None = None,
) -> float:
    """Return 1.0 if normalized prediction matches any reference exactly."""
    pred = normalize_text(prediction, ignore_case, ignore_punctuation, regexes_to_ignore)
    refs = reference if isinstance(reference, list) else [reference]
    return 1.0 if pred in [normalize_text(r, ignore_case, ignore_punctuation, regexes_to_ignore) for r in refs] else 0.0


def relaxed_accuracy(prediction: str, reference: str | list[str]) -> float:
    """ChartQA metric: exact match or 5% numeric tolerance. Extracts from 'Final Answer:' format."""
    def extract_answer(text: str) -> str:
        if match := re.search(r"Final Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE):
            return match.group(1).strip()
        return text.strip()

    def parse_number(s: str) -> float | None:
        try:
            return float(s.replace(",", "").replace("%", "").replace("$", "").strip())
        except ValueError:
            return None

    pred = extract_answer(prediction)
    for ref in (reference if isinstance(reference, list) else [reference]):
        ref_clean = str(ref).strip()
        if pred.lower() == ref_clean.lower():
            return 1.0
        pred_num, ref_num = parse_number(pred), parse_number(ref_clean)
        if pred_num is not None and ref_num is not None:
            if ref_num == 0:
                if pred_num == 0:
                    return 1.0
            elif abs(pred_num - ref_num) / abs(ref_num) <= 0.05:
                return 1.0
    return 0.0


def anywhere_accuracy(prediction: str, reference: str | list[str]) -> float:
    """Return 1.0 if any reference appears as substring in prediction (case-insensitive)."""
    pred_lower = prediction.lower()
    refs = reference if isinstance(reference, list) else [reference]
    return 1.0 if any(str(ref).lower().strip() in pred_lower for ref in refs) else 0.0


def apply_filter(text: str, filter_config: dict[str, Any]) -> str:
    """Apply regex or take_first filter to extract answer from generation."""
    func = filter_config.get("function", "")
    if func == "regex":
        if match := re.search(filter_config.get("regex_pattern", ""), text):
            group = filter_config.get("group_select", 0)
            if group == -1:
                return match.groups()[-1] if match.groups() else match.group(0)
            try:
                return match.group(group)
            except IndexError:
                return match.group(0)
    return text


def compute_metrics(instances: list[Instance], config: TaskConfig) -> dict[str, float]:
    """Compute configured metrics over all instances, returning averages."""
    metrics: dict[str, list[float]] = {}
    metric_configs = config.metric_list or [{"metric": "exact_match"}]

    for inst in instances:
        response = inst.response or ""
        if config.filter_list:
            for group in config.filter_list:
                for cfg in group.get("filter", []):
                    response = apply_filter(response, cfg)

        for cfg in metric_configs:
            name = cfg.get("metric", "exact_match")
            # Normalize function refs like "!function utils.exact_match"
            if isinstance(name, str):
                for key in ("exact_match", "relaxed_accuracy", "anywhere_accuracy"):
                    if key in name:
                        name = key
                        break

            if name == "exact_match":
                score = exact_match(response, inst.target, cfg.get("ignore_case", False),
                                    cfg.get("ignore_punctuation", False), cfg.get("regexes_to_ignore"))
            elif name == "relaxed_accuracy":
                score = relaxed_accuracy(response, inst.target)
            elif name == "anywhere_accuracy":
                score = anywhere_accuracy(response, inst.target)
            else:
                score = exact_match(response, inst.target, ignore_case=True)

            metrics.setdefault(name, []).append(score)

    return {name: sum(scores) / len(scores) for name, scores in metrics.items() if scores}


# ============================================================================
# LocalCompletionsAPI - Compatible interface for tests
# ============================================================================

class LocalCompletionsAPI:
    """OpenAI-compatible completions API client (lm_eval-compatible interface)."""

    def __init__(
        self, base_url: str, model: str = "gpt-3.5-turbo",
        num_concurrent: int = 1, max_retries: int = 3,
        seed: int = 1234, timeout: int = 300, **kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._seed = seed
        self._max_gen_toks = 256
        self._concurrent = num_concurrent
        self.max_retries = max_retries
        self.timeout = timeout

    @property
    def header(self) -> dict[str, str]:
        """Auth header if api_key is set."""
        return {"Authorization": f"Bearer {k}"} if (k := getattr(self, "api_key", "")) else {}

    def _create_payload(
        self, messages: str | list[str], generate: bool = False,
        gen_kwargs: dict[str, Any] | None = None, seed: int = 1234, eos: str | None = None, **kwargs: Any,
    ) -> dict[str, Any]:
        """Build completions API payload for generate or logprobs mode."""
        gen_kwargs = gen_kwargs or {}
        if generate:
            gen_kwargs.pop("do_sample", None)
            max_tokens = gen_kwargs.pop("max_tokens", None) or gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None))
            if eos and eos not in stop:
                stop.append(eos)
            return {"prompt": messages, "model": self.model, "max_tokens": max_tokens,
                    "temperature": gen_kwargs.pop("temperature", 0), "stop": stop or None, "seed": seed, **gen_kwargs}
        return {"model": self.model, "prompt": messages, "temperature": 0,
                "max_tokens": 1, "logprobs": 1, "seed": seed, "echo": True}

    def model_call(
        self, messages: str | list[str], generate: bool = True,
        gen_kwargs: dict[str, Any] | None = None, **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Synchronous API request."""
        import requests
        payload = {k: v for k, v in self._create_payload(
            messages, generate, copy.deepcopy(gen_kwargs) if gen_kwargs else {}, self._seed, **kwargs
        ).items() if v is not None}
        try:
            resp = requests.post(self.base_url, json=payload, headers=self.header)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"Request failed: {e}")
            return None

    async def get_batched_requests(
        self, requests: list[str],
        generate: bool = True, gen_kwargs: dict[str, Any] | None = None, **kwargs: Any,
    ) -> list[list[str] | list[tuple[float, bool]]]:
        """Async batched requests with concurrency control."""
        gen_kwargs = copy.deepcopy(gen_kwargs) if gen_kwargs else {}
        semaphore = asyncio.Semaphore(self._concurrent)

        async def make_single(session: aiohttp.ClientSession, msg: str) -> dict[str, Any] | None:
            payload = {k: v for k, v in self._create_payload(
                msg, generate, copy.deepcopy(gen_kwargs), self._seed, **kwargs
            ).items() if v is not None}
            async with semaphore:
                try:
                    async with session.post(self.base_url, json=payload, headers=self.header) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except Exception as e:
                    log.warning(f"Async request failed: {e}")
                    return None

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self._concurrent),
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            responses = await asyncio.gather(*[make_single(session, msg) for msg in requests])

        return [[self._parse_generation(r)] if generate else [(0.0, True)] for r in responses]

    @staticmethod
    def _parse_generation(response: dict[str, Any] | None) -> str:
        """Extract text from completions response."""
        if not response or not (choices := response.get("choices")):
            return ""
        return choices[0].get("text", "")

    @staticmethod
    def parse_generations(outputs: dict | list[dict], **kwargs: Any) -> list[str]:
        """Extract all texts from completions response(s)."""
        if not isinstance(outputs, list):
            outputs = [outputs]
        return [choice.get("text", "") for out in outputs if out and "choices" in out for choice in out["choices"]]


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def download_dataset(dataset_path: str, dataset_name: str | None = None) -> None:
    """Pre-download and cache dataset. Call before timed evaluation."""
    log.info(f"Pre-downloading: {dataset_path}")
    datasets.load_dataset(path=dataset_path, name=dataset_name)


def load_all_docs(dataset_path: str, dataset_name: str | None, limit: int | None = None) -> list[dict]:
    """Load docs from dataset. Streams if limit to avoid full download; loads all splits otherwise."""
    if limit:
        for split in ["test", "validation", "train"]:
            try:
                return list(datasets.load_dataset(dataset_path, dataset_name, split=split, streaming=True).take(limit))
            except ValueError:
                continue
        raise ValueError(f"No valid split found in {dataset_path}")

    dataset_dict = datasets.load_dataset(path=dataset_path, name=dataset_name)
    return [doc for split in dataset_dict for doc in dataset_dict[split]]


async def evaluate_task(
    task_path: Path, api_config: APIConfig, num_fewshot: int | None = None,
    limit: int | None = None, seed: int = 42,
) -> tuple[EvalResult, float]:
    """Load task config, build instances, run generation, compute metrics. Returns (result, eval_time)."""
    config = TaskConfig.from_yaml(task_path)
    log.info(f"Evaluating task: {config.task} (multimodal: {config.is_multimodal})")
    if num_fewshot is not None:
        config.num_fewshot = num_fewshot

    docs = load_all_docs(config.dataset_path, config.dataset_name, limit)
    rng = random.Random(seed)
    fewshot_context = build_fewshot_context(config, get_fewshot_examples(docs, config.num_fewshot, rng))

    instances = build_instances(config, docs, fewshot_context)
    log.info(f"Built {len(instances)} instances")

    start_time = time.perf_counter()
    instances = await run_generation(instances, api_config, is_multimodal=config.is_multimodal)
    eval_time = time.perf_counter() - start_time

    return EvalResult(task=config.task, metrics=compute_metrics(instances, config), num_samples=len(instances)), eval_time


async def evaluate_tasks(
    task_paths: list[Path], api_config: APIConfig, num_fewshot: int | None = None,
    limit: int | None = None, seed: int = 42,
) -> tuple[dict[str, tuple[EvalResult, float]], float]:
    """Evaluate multiple tasks sequentially, returning (result, time) per task and total time."""
    results = {}
    total_eval_time = 0.0
    for path in task_paths:
        try:
            result, eval_time = await evaluate_task(path, api_config, num_fewshot, limit, seed)
            results[result.task] = (result, eval_time)
            total_eval_time += eval_time
            log.info(f"Task {result.task}: {result.metrics} ({eval_time:.2f}s)")
        except Exception as e:
            log.error(f"Error evaluating {path}: {e}")
            import traceback
            traceback.print_exc()
    return results, total_eval_time


def find_task_configs(tasks_dir: Path, task_names: list[str]) -> list[Path]:
    """Locate YAML configs by name: direct path, exact match, or first glob match."""
    paths = []
    for name in task_names:
        if Path(name).exists():
            paths.append(Path(name))
            continue
        matches = list(tasks_dir.rglob(f"{name}.yaml")) + list(tasks_dir.rglob(f"{name}/*.yaml"))
        if matches:
            paths.append(next((m for m in matches if m.stem == name), matches[0]))
        else:
            log.warning(f"Task not found: {name}")
    return paths


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI: evaluate tasks against an OpenAI-compatible API."""
    import argparse
    parser = argparse.ArgumentParser(description="Minimal LM Evaluation Harness")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base_url", type=str, required=True, help="API base URL (chat completions endpoint)")
    parser.add_argument("--api_key", type=str, default="", help="API key")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--num_concurrent", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks_dir", type=str, default="lm_eval/tasks", help="Tasks directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    task_paths = find_task_configs(Path(args.tasks_dir), [t.strip() for t in args.tasks.split(",")])
    if not task_paths:
        log.error("No valid tasks found")
        return 1

    api_config = APIConfig(base_url=args.base_url, model=args.model, api_key=args.api_key, num_concurrent=args.num_concurrent)
    results, total_eval_time = asyncio.run(evaluate_tasks(task_paths, api_config, args.num_fewshot, args.limit, args.seed))

    output = {
        "results": {
            name: {"metrics": r.metrics, "num_samples": r.num_samples, "evaluation_time_seconds": str(t)}
            for name, (r, t) in results.items()
        },
        "config": {"model": args.model, "num_fewshot": args.num_fewshot, "limit": args.limit},
        "total_evaluation_time_seconds": str(total_eval_time),
    }
    print(json.dumps(output, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
    return 0


if __name__ == "__main__":
    exit(main())
