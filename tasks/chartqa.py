"""
ChartQA evaluation - multimodal chart understanding.

Defines:
- samples(): generator yielding ((prompt, images), target) pairs
- score(): relaxed matching with 5% numeric tolerance
- chartqa: Task instance for registration
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import datasets

from core import Sample, Task

# Extracts answer after "FINAL ANSWER:" up to newline or end (prompt instructs model to use this)
# Non-greedy (.+?) stops at first newline to avoid capturing extra text
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# Strip currency/percent symbols for numeric comparison: "$1,234%" -> "1234"
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


def samples() -> Iterator[Sample]:
    """Generate ChartQA samples: ((prompt, [image]), target)."""
    for split in ["test", "val", "train"]:
        ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split=split, streaming=True)
        for doc in ds:
            label = doc["label"]
            target = label[0] if isinstance(label, list) else str(label)
            yield Sample(
                prompt=(_format_chartqa_prompt(doc["query"]), [doc["image"]]),
                target=target,
            )


def score(response: str, target: str) -> float:
    """Score ChartQA response with relaxed matching (5% numeric tolerance)."""
    return _relaxed_match(response, target)


# Task instance for registration
chartqa = Task(
    name="chartqa",
    samples=samples,
    score=score,
)
