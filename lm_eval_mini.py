#!/usr/bin/env python3
"""
Minimal LM Evaluation Harness - A clean, typed implementation under 1k lines.

Supports:
- YAML task configs with jinja2 templates
- Async HTTP requests to OpenAI-compatible APIs
- generate_until and multiple_choice output types
- Core metrics: exact_match, acc, acc_norm
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import aiohttp
import datasets
import jinja2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# Types
# ============================================================================

OutputType = Literal["generate_until", "multiple_choice", "loglikelihood"]


@dataclass
class TaskConfig:
    """Configuration loaded from a YAML task file."""
    task: str
    dataset_path: str
    dataset_name: str | None = None
    test_split: str = "test"
    training_split: str | None = None
    fewshot_split: str | None = None
    num_fewshot: int = 0
    output_type: OutputType = "generate_until"
    doc_to_text: str | None = None
    doc_to_target: str | None = None
    doc_to_choice: str | list[str] | None = None
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    metric_list: list[dict[str, Any]] = field(default_factory=list)
    filter_list: list[dict[str, Any]] | None = None
    process_docs: Callable[[Any], Any] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> TaskConfig:
        """Load a task config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle includes
        if "include" in data:
            include_path = path.parent / data.pop("include")
            base = cls.from_yaml(include_path)
            # Merge data over base
            for key, value in data.items():
                setattr(base, key, value)
            return base

        # Set task name from file if not specified
        if "task" not in data:
            data["task"] = path.stem

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Instance:
    """A single evaluation instance."""
    doc: dict[str, Any]
    doc_id: int
    prompt: str
    request_type: OutputType
    target: str | int | list[str]
    choices: list[str] | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    response: str | None = None
    logprobs: list[tuple[float, bool]] | None = None


@dataclass
class EvalResult:
    """Results for a single task."""
    task: str
    metrics: dict[str, float]
    num_samples: int
    samples: list[dict[str, Any]] | None = None


# ============================================================================
# Template Engine
# ============================================================================

_jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)


def render_template(template: str, doc: dict[str, Any]) -> str:
    """Render a jinja2 template with the document context."""
    try:
        return _jinja_env.from_string(template).render(**doc)
    except jinja2.UndefinedError as e:
        log.warning(f"Template error: {e}")
        return template


def resolve_field(doc: dict[str, Any], field_spec: str | list | int | None) -> Any:
    """Resolve a field specification against a document."""
    if field_spec is None:
        return None
    if isinstance(field_spec, int):
        return field_spec
    if isinstance(field_spec, list):
        return field_spec
    # Check if it's a direct field reference
    if field_spec in doc:
        return doc[field_spec]
    # Otherwise treat as jinja2 template
    result = render_template(field_spec, doc)
    # Try to parse as int or list if applicable
    if result.isdigit():
        return int(result)
    if result.startswith("[") and result.endswith("]"):
        try:
            return json.loads(result.replace("'", '"'))
        except json.JSONDecodeError:
            pass
    return result


# ============================================================================
# Task Loading and Instance Building
# ============================================================================

def load_dataset_for_task(config: TaskConfig) -> datasets.Dataset:
    """Load the HuggingFace dataset for a task."""
    ds = datasets.load_dataset(
        path=config.dataset_path,
        name=config.dataset_name,
    )
    split = ds[config.test_split]
    if config.process_docs:
        split = config.process_docs(split)
    return split


def get_fewshot_examples(
    config: TaskConfig,
    dataset: datasets.DatasetDict,
    num_fewshot: int,
    rng: random.Random,
) -> list[dict]:
    """Sample few-shot examples from the training/fewshot split."""
    if num_fewshot == 0:
        return []

    split_name = config.fewshot_split or config.training_split
    if not split_name or split_name not in dataset:
        return []

    fewshot_docs = list(dataset[split_name])
    return rng.sample(fewshot_docs, min(num_fewshot, len(fewshot_docs)))


def build_fewshot_context(
    config: TaskConfig,
    examples: list[dict],
) -> str:
    """Build the few-shot context string."""
    if not examples:
        return ""

    parts = []
    for doc in examples:
        text = resolve_field(doc, config.doc_to_text)
        target = resolve_field(doc, config.doc_to_target)

        # For multiple choice, get the actual answer text
        if config.doc_to_choice and isinstance(target, int):
            choices = resolve_field(doc, config.doc_to_choice)
            target = choices[target]

        parts.append(f"{text}{config.target_delimiter}{target}")

    return config.fewshot_delimiter.join(parts) + config.fewshot_delimiter


def build_instances(
    config: TaskConfig,
    docs: list[dict],
    fewshot_context: str = "",
    limit: int | None = None,
) -> list[Instance]:
    """Build evaluation instances from documents."""
    instances = []
    docs_to_use = docs[:limit] if limit else docs

    for doc_id, doc in enumerate(docs_to_use):
        text = resolve_field(doc, config.doc_to_text)
        target = resolve_field(doc, config.doc_to_target)
        prompt = fewshot_context + str(text)

        if config.output_type == "multiple_choice":
            choices = resolve_field(doc, config.doc_to_choice)
            instances.append(Instance(
                doc=doc,
                doc_id=doc_id,
                prompt=prompt,
                request_type="loglikelihood",
                target=target,
                choices=choices,
            ))
        else:  # generate_until
            instances.append(Instance(
                doc=doc,
                doc_id=doc_id,
                prompt=prompt,
                request_type="generate_until",
                target=str(target) if not isinstance(target, list) else target,
                generation_kwargs=config.generation_kwargs.copy(),
            ))

    return instances


# ============================================================================
# HTTP Client for OpenAI-compatible APIs
# ============================================================================

@dataclass
class APIConfig:
    """Configuration for the API client."""
    base_url: str
    model: str
    api_key: str = ""
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 1234
    num_concurrent: int = 8
    timeout: int = 300
    max_retries: int = 3


def handle_stop_sequences(
    until: list[str] | str | None,
    eos: str | None = None,
) -> list[str]:
    """Process stop sequences into a clean list."""
    if until is None:
        until = []
    elif isinstance(until, str):
        until = [until]
    else:
        until = list(until)

    if eos and eos not in until:
        until.append(eos)

    return [s for s in until if s]


class LocalCompletionsAPI:
    """
    OpenAI-compatible completions API client.
    Compatible interface with the original lm_eval implementation.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "gpt-3.5-turbo",
        tokenizer_backend: str | None = None,
        num_concurrent: int = 1,
        batch_size: int = 1,
        max_retries: int = 3,
        seed: int = 1234,
        timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._seed = seed
        self._max_gen_toks = 256
        self._concurrent = num_concurrent
        self._batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

    @property
    def header(self) -> dict[str, str]:
        """Return headers for API requests."""
        api_key = getattr(self, "api_key", "")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _create_payload(
        self,
        messages: str | list[str],
        generate: bool = False,
        gen_kwargs: dict[str, Any] | None = None,
        seed: int = 1234,
        eos: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create the request payload."""
        gen_kwargs = gen_kwargs or {}

        if generate:
            gen_kwargs.pop("do_sample", None)
            max_tokens = gen_kwargs.pop("max_tokens", None) or gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop if stop else None,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    def model_call(
        self,
        messages: str | list[str],
        generate: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Make a synchronous API request."""
        import copy
        import requests

        gen_kwargs = copy.deepcopy(gen_kwargs) if gen_kwargs else {}
        payload = self._create_payload(
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )

        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.header,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.warning(f"Request failed: {e}")
            return None

    async def get_batched_requests(
        self,
        requests: list[str],
        cache_keys: list[tuple[str, ...]],
        generate: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[list[str] | list[tuple[float, bool]]]:
        """Make batched async API requests."""
        import copy

        gen_kwargs = copy.deepcopy(gen_kwargs) if gen_kwargs else {}
        results: list[Any] = []

        connector = aiohttp.TCPConnector(limit=self._concurrent)
        timeout_cfg = aiohttp.ClientTimeout(total=self.timeout)
        semaphore = asyncio.Semaphore(self._concurrent)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
            async def make_single_request(msg: str) -> dict[str, Any] | None:
                payload = self._create_payload(
                    msg,
                    generate=generate,
                    gen_kwargs=copy.deepcopy(gen_kwargs),
                    seed=self._seed,
                    **kwargs,
                )
                payload = {k: v for k, v in payload.items() if v is not None}

                async with semaphore:
                    try:
                        async with session.post(
                            self.base_url,
                            json=payload,
                            headers=self.header,
                        ) as response:
                            response.raise_for_status()
                            return await response.json()
                    except Exception as e:
                        log.warning(f"Async request failed: {e}")
                        return None

            tasks = [make_single_request(msg) for msg in requests]
            responses = await asyncio.gather(*tasks)

            for resp in responses:
                if generate:
                    results.append([self._parse_generation(resp)])
                else:
                    results.append([self._parse_logprobs(resp)])

        return results

    @staticmethod
    def _parse_generation(response: dict[str, Any] | None) -> str:
        """Parse generation from response."""
        if not response or "choices" not in response:
            return ""
        choices = response["choices"]
        if not choices:
            return ""
        return choices[0].get("text", "")

    @staticmethod
    def _parse_logprobs(response: dict[str, Any] | None) -> tuple[float, bool]:
        """Parse logprobs from response."""
        if not response or "choices" not in response:
            return (0.0, False)
        return (0.0, True)  # Simplified

    @staticmethod
    def parse_generations(outputs: dict | list[dict], **kwargs: Any) -> list[str]:
        """Parse generations from API response."""
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            if out and "choices" in out:
                for choice in out["choices"]:
                    res.append(choice.get("text", ""))
        return res


def create_completion_payload(
    prompt: str | list[str],
    config: APIConfig,
    gen_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the payload for a completions API request."""
    gen_kwargs = gen_kwargs or {}

    max_tokens = gen_kwargs.get("max_tokens") or gen_kwargs.get("max_gen_toks", config.max_tokens)
    temperature = gen_kwargs.get("temperature", config.temperature)
    stop = gen_kwargs.get("until", [])

    # Filter out None and empty values from stop sequences
    if isinstance(stop, list):
        stop = [s for s in stop if s]

    payload = {
        "model": config.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": config.seed,
    }

    if stop:
        payload["stop"] = stop

    return payload


def create_logprobs_payload(
    prompt: str,
    config: APIConfig,
) -> dict[str, Any]:
    """Create the payload for a logprobs request."""
    return {
        "model": config.model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "echo": True,
        "seed": config.seed,
    }


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """Make a single API request with retries."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.ok:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        log.warning(f"Request failed (attempt {attempt + 1}): {error_text}")
        except Exception as e:
            log.warning(f"Request error (attempt {attempt + 1}): {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    return None


def parse_generation(response: dict[str, Any]) -> str:
    """Parse a generation response."""
    if not response or "choices" not in response:
        return ""
    choices = response["choices"]
    if not choices:
        return ""
    return choices[0].get("text", "")


def parse_logprobs(
    response: dict[str, Any],
    context_len: int,
) -> tuple[float, bool]:
    """Parse logprobs from a response."""
    if not response or "choices" not in response:
        return (0.0, False)

    choice = response["choices"][0]
    token_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
    top_logprobs = choice.get("logprobs", {}).get("top_logprobs", [])

    if not token_logprobs or context_len >= len(token_logprobs):
        return (0.0, False)

    # Sum logprobs after context
    logprob_sum = sum(token_logprobs[context_len:])

    # Check if greedy
    is_greedy = True
    for i in range(context_len, len(token_logprobs)):
        if i < len(top_logprobs) and top_logprobs[i]:
            tok_lp = token_logprobs[i]
            max_lp = max(top_logprobs[i].values())
            if tok_lp < max_lp:
                is_greedy = False
                break

    return (logprob_sum, is_greedy)


async def run_generate_until(
    instances: list[Instance],
    config: APIConfig,
) -> list[Instance]:
    """Run generate_until requests for all instances."""
    if not instances:
        return instances

    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    semaphore = asyncio.Semaphore(config.num_concurrent)
    connector = aiohttp.TCPConnector(limit=config.num_concurrent)
    timeout = aiohttp.ClientTimeout(total=config.timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for inst in instances:
            payload = create_completion_payload(
                inst.prompt,
                config,
                inst.generation_kwargs,
            )
            tasks.append(make_request(
                session, config.base_url, payload, headers, semaphore, config.max_retries
            ))

        responses = await asyncio.gather(*tasks)

        for inst, response in zip(instances, responses):
            inst.response = parse_generation(response) if response else ""

    return instances


async def run_loglikelihood(
    instances: list[Instance],
    config: APIConfig,
) -> list[Instance]:
    """Run loglikelihood requests for multiple choice instances."""
    if not instances:
        return instances

    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    semaphore = asyncio.Semaphore(config.num_concurrent)
    connector = aiohttp.TCPConnector(limit=config.num_concurrent)
    timeout = aiohttp.ClientTimeout(total=config.timeout)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for inst in instances:
            if not inst.choices:
                continue

            # Make a request for each choice
            choice_logprobs = []
            for choice in inst.choices:
                prompt_with_choice = f"{inst.prompt}{choice}"
                payload = create_logprobs_payload(prompt_with_choice, config)

                response = await make_request(
                    session, config.base_url, payload, headers, semaphore, config.max_retries
                )

                if response:
                    # Estimate context length (chars in prompt / ~4 for avg token length)
                    # This is approximate - real impl would use tokenizer
                    context_len = len(inst.prompt) // 4
                    logprob, is_greedy = parse_logprobs(response, context_len)
                    choice_logprobs.append((logprob, is_greedy))
                else:
                    choice_logprobs.append((float("-inf"), False))

            inst.logprobs = choice_logprobs

    return instances


# ============================================================================
# Metrics
# ============================================================================

def normalize_text(
    text: str,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    regexes_to_ignore: list[str] | None = None,
) -> str:
    """Normalize text for comparison."""
    if regexes_to_ignore:
        for pattern in regexes_to_ignore:
            text = re.sub(pattern, "", text)
    if ignore_case:
        text = text.lower()
    if ignore_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def exact_match(
    prediction: str,
    reference: str | list[str],
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    regexes_to_ignore: list[str] | None = None,
) -> float:
    """Compute exact match metric."""
    pred_norm = normalize_text(prediction, ignore_case, ignore_punctuation, regexes_to_ignore)

    if isinstance(reference, list):
        refs_norm = [normalize_text(r, ignore_case, ignore_punctuation, regexes_to_ignore) for r in reference]
        return 1.0 if pred_norm in refs_norm else 0.0

    ref_norm = normalize_text(reference, ignore_case, ignore_punctuation, regexes_to_ignore)
    return 1.0 if pred_norm == ref_norm else 0.0


def apply_filter(text: str, filter_config: dict[str, Any]) -> str:
    """Apply a filter to extract answer from generation."""
    func = filter_config.get("function", "")

    if func == "regex":
        pattern = filter_config.get("regex_pattern", "")
        group_select = filter_config.get("group_select", 0)
        match = re.search(pattern, text)
        if match:
            if group_select == -1:
                # Return the last group or full match
                groups = match.groups()
                return groups[-1] if groups else match.group(0)
            return match.group(group_select) if group_select <= len(match.groups()) else match.group(0)
    elif func == "take_first":
        return text  # Already the first

    return text


def compute_generate_until_metrics(
    instances: list[Instance],
    config: TaskConfig,
) -> dict[str, float]:
    """Compute metrics for generate_until tasks."""
    metrics: dict[str, list[float]] = {}

    # Get metric configurations
    metric_configs = config.metric_list or [{"metric": "exact_match"}]

    for inst in instances:
        response = inst.response or ""

        # Apply filters if specified
        if config.filter_list:
            for filter_group in config.filter_list:
                for filter_cfg in filter_group.get("filter", []):
                    response = apply_filter(response, filter_cfg)

        for metric_cfg in metric_configs:
            metric_name = metric_cfg.get("metric", "exact_match")

            if metric_name == "exact_match":
                score = exact_match(
                    response,
                    inst.target,
                    ignore_case=metric_cfg.get("ignore_case", False),
                    ignore_punctuation=metric_cfg.get("ignore_punctuation", False),
                    regexes_to_ignore=metric_cfg.get("regexes_to_ignore"),
                )
                metrics.setdefault(metric_name, []).append(score)

    # Aggregate metrics
    return {name: sum(scores) / len(scores) for name, scores in metrics.items() if scores}


def compute_multiple_choice_metrics(
    instances: list[Instance],
) -> dict[str, float]:
    """Compute metrics for multiple choice tasks."""
    acc_scores = []
    acc_norm_scores = []

    for inst in instances:
        if not inst.logprobs or not inst.choices:
            continue

        lls = [lp[0] for lp in inst.logprobs]

        # Get the gold answer index
        gold = inst.target
        if isinstance(gold, str):
            # Try to find in choices
            if gold in inst.choices:
                gold = inst.choices.index(gold)
            else:
                # Try as single letter (A, B, C, D)
                try:
                    gold = ord(gold.upper()) - ord("A")
                except (TypeError, ValueError):
                    gold = 0

        if not isinstance(gold, int):
            gold = 0

        # Accuracy
        pred = int(np.argmax(lls))
        acc_scores.append(1.0 if pred == gold else 0.0)

        # Normalized accuracy (by choice length)
        choice_lens = [len(c) for c in inst.choices]
        lls_norm = [ll / l if l > 0 else ll for ll, l in zip(lls, choice_lens)]
        pred_norm = int(np.argmax(lls_norm))
        acc_norm_scores.append(1.0 if pred_norm == gold else 0.0)

    results = {}
    if acc_scores:
        results["acc"] = sum(acc_scores) / len(acc_scores)
    if acc_norm_scores:
        results["acc_norm"] = sum(acc_norm_scores) / len(acc_norm_scores)

    return results


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

async def evaluate_task(
    task_path: Path,
    api_config: APIConfig,
    num_fewshot: int | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> EvalResult:
    """Evaluate a single task."""
    # Load task config
    config = TaskConfig.from_yaml(task_path)
    log.info(f"Evaluating task: {config.task}")

    # Override num_fewshot if specified
    if num_fewshot is not None:
        config.num_fewshot = num_fewshot

    # Load dataset
    dataset = datasets.load_dataset(
        path=config.dataset_path,
        name=config.dataset_name,
    )

    # Get test docs
    test_docs = list(dataset[config.test_split])
    if limit:
        test_docs = test_docs[:limit]

    # Build few-shot context
    rng = random.Random(seed)
    fewshot_examples = get_fewshot_examples(config, dataset, config.num_fewshot, rng)
    fewshot_context = build_fewshot_context(config, fewshot_examples)

    # Build instances
    instances = build_instances(config, test_docs, fewshot_context)
    log.info(f"Built {len(instances)} instances")

    # Run evaluation based on output type
    if config.output_type == "generate_until":
        instances = await run_generate_until(instances, api_config)
        metrics = compute_generate_until_metrics(instances, config)
    else:  # multiple_choice
        instances = await run_loglikelihood(instances, api_config)
        metrics = compute_multiple_choice_metrics(instances)

    return EvalResult(
        task=config.task,
        metrics=metrics,
        num_samples=len(instances),
    )


async def evaluate_tasks(
    task_paths: list[Path],
    api_config: APIConfig,
    num_fewshot: int | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, EvalResult]:
    """Evaluate multiple tasks."""
    results = {}
    for path in task_paths:
        try:
            result = await evaluate_task(path, api_config, num_fewshot, limit, seed)
            results[result.task] = result
            log.info(f"Task {result.task}: {result.metrics}")
        except Exception as e:
            log.error(f"Error evaluating {path}: {e}")
    return results


def find_task_configs(tasks_dir: Path, task_names: list[str]) -> list[Path]:
    """Find task config files by name."""
    paths = []
    for name in task_names:
        # Try direct path
        if Path(name).exists():
            paths.append(Path(name))
            continue

        # Search in tasks directory
        matches = list(tasks_dir.rglob(f"{name}.yaml")) + list(tasks_dir.rglob(f"{name}/*.yaml"))
        if matches:
            # Prefer exact match
            for m in matches:
                if m.stem == name:
                    paths.append(m)
                    break
            else:
                paths.append(matches[0])
        else:
            log.warning(f"Task not found: {name}")

    return paths


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Minimal LM Evaluation Harness")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base_url", type=str, required=True, help="API base URL")
    parser.add_argument("--api_key", type=str, default="", help="API key")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--num_concurrent", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks_dir", type=str, default="lm_eval/tasks", help="Tasks directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Find task configs
    tasks_dir = Path(args.tasks_dir)
    task_names = [t.strip() for t in args.tasks.split(",")]
    task_paths = find_task_configs(tasks_dir, task_names)

    if not task_paths:
        log.error("No valid tasks found")
        return 1

    # Configure API
    api_config = APIConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        num_concurrent=args.num_concurrent,
    )

    # Run evaluation
    results = asyncio.run(evaluate_tasks(
        task_paths,
        api_config,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        seed=args.seed,
    ))

    # Output results
    output = {
        "results": {name: {"metrics": r.metrics, "num_samples": r.num_samples} for name, r in results.items()},
        "config": {
            "model": args.model,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
        }
    }

    print(json.dumps(output, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
