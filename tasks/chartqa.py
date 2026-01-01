"""ChartQA task - pure logic only.

This module provides data loading, prompt formatting, and scoring for ChartQA.
It doesn't run anything - the execution loop is in tinyeval.py.
"""

from __future__ import annotations

import re

import datasets

# Task metadata
NAME = "chartqa"

# No stop sequences needed for ChartQA
STOP = None

# Pre-compiled regex patterns
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_NUMERIC_CLEAN_RE = re.compile(r"[$,%]")

# Pre-compiled regex patterns for normalization
_NORMALIZE_CURRENCY_RE = re.compile(r"[$,]")
_NORMALIZE_THOUGHT_RE = re.compile(r"(?s).*#### ")
_NORMALIZE_END_RE = re.compile(r"\.$")


def load(limit: int | None = None) -> list:
    """Load ChartQA dataset."""
    # Prefer test, then validation, then train
    for split in ["test", "validation", "train"]:
        try:
            ds = datasets.load_dataset(
                "HuggingFaceM4/ChartQA", split=split, streaming=True
            )
            return list(ds.take(limit) if limit else ds)
        except ValueError:
            continue
    raise ValueError("No valid split found in HuggingFaceM4/ChartQA")


def prompt(doc: dict) -> tuple[str, list]:
    """Format a single document into a (text, images) tuple for multimodal."""
    text = (
        f"<image>You are provided a chart image and will be asked a question. "
        f"You have to think through your answer and provide a step-by-step solution. "
        f'Once you have the solution, write the final answer in at most a few words at the end with the phrase "FINAL ANSWER:". '
        f"The question is: {doc['query']}\n"
        f"Let's think step by step."
    )
    return (text, [doc["image"]])


def score(responses: list[str], docs: list) -> dict:
    """Score responses against documents. Returns metrics dict."""
    targets = [
        d["label"][0] if isinstance(d["label"], list) else str(d["label"]) for d in docs
    ]

    # Relaxed accuracy (exact match or 5% numeric tolerance)
    relaxed_correct = sum(_relaxed_match(r, t) for r, t in zip(responses, targets))

    # Exact match (normalized)
    exact_correct = sum(
        _normalize(r) == _normalize(t) for r, t in zip(responses, targets)
    )

    return {
        "exact_match": exact_correct / len(docs),
        "relaxed_accuracy": relaxed_correct / len(docs),
    }


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


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = _NORMALIZE_CURRENCY_RE.sub("", text)
    text = _NORMALIZE_THOUGHT_RE.sub("", text)
    text = _NORMALIZE_END_RE.sub("", text)
    return text.lower().strip()
