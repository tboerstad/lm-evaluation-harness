"""
ChartQA evaluation - multimodal chart understanding.

Responsibilities:
- Load ChartQA dataset (test → val → train, stops at limit)
- Format prompts with image and query
- Extract "FINAL ANSWER:" from responses
- Compute exact_match + relaxed_accuracy (5% tolerance)
"""

from __future__ import annotations

import logging
import re

import datasets

from core import APIConfig, _normalize, run_task
from result_types import TaskResult

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for _relaxed_match
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
    # Extract "FINAL ANSWER: X"
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


async def eval_chartqa(config: APIConfig, max_samples: int | None = None) -> TaskResult:
    """
    Evaluate ChartQA - multimodal chart understanding.

    Returns TaskResult with relaxed_accuracy, num_samples, elapsed.
    """
    docs = []
    for split in ["test", "val", "train"]:
        ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split=split, streaming=True)
        for doc in ds:
            docs.append(doc)
            if max_samples and len(docs) >= max_samples:
                break
        if max_samples and len(docs) >= max_samples:
            break

    targets = [
        d["label"][0] if isinstance(d["label"], list) else str(d["label"]) for d in docs
    ]

    responses, elapsed = await run_task(
        "chartqa",
        config,
        docs,
        lambda d: (_format_chartqa_prompt(d["query"]), [d["image"]]),
    )

    correct = sum(_relaxed_match(r, t) for r, t in zip(responses, targets))

    metrics = {
        "exact_match": sum(
            _normalize(r) == _normalize(t) for r, t in zip(responses, targets)
        )
        / len(docs),
        "relaxed_accuracy": correct / len(docs),
    }
    logger.info("chartqa: %s (%.2fs)", metrics, elapsed)

    return {
        "task": "chartqa",
        "metrics": metrics,
        "num_samples": len(docs),
        "elapsed": round(elapsed, 2),
    }
