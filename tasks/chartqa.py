"""ChartQA task - multimodal chart understanding."""

from __future__ import annotations

import re
import time

import datasets

from tinyeval import APIConfig, _normalize, complete


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
    if match := re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE):
        pred = match.group(1).strip()
    else:
        pred = response.strip()

    if pred.lower() == target.lower():
        return 1.0

    try:
        pred_n = float(re.sub(r"[$,%]", "", pred))
        target_n = float(re.sub(r"[$,%]", "", target))
        if target_n == 0:
            return 1.0 if pred_n == 0 else 0.0
        if abs(pred_n - target_n) / abs(target_n) <= 0.05:
            return 1.0
    except ValueError:
        pass

    return 0.0


async def eval_chartqa(config: APIConfig, limit: int | None = None) -> dict:
    """
    Evaluate ChartQA - multimodal chart understanding.

    Returns dict with relaxed_accuracy, num_samples, time_seconds.
    """
    # Load
    for split in ["test", "validation", "train"]:
        try:
            ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split=split, streaming=True)
            docs = list(ds.take(limit) if limit else ds)
            break
        except ValueError:
            continue
    else:
        raise ValueError("No valid split found in HuggingFaceM4/ChartQA")

    # Format prompts with images
    prompts: list[tuple[str, list]] = [(_format_chartqa_prompt(d["query"]), [d["image"]]) for d in docs]
    targets = [d["label"][0] if isinstance(d["label"], list) else str(d["label"]) for d in docs]

    # Run inference
    print(f"Evaluating: chartqa ({len(docs)} samples, multimodal)")
    t0 = time.perf_counter()
    responses = await complete(prompts, config, max_tokens=512)
    elapsed = time.perf_counter() - t0

    # Score
    correct = sum(_relaxed_match(r, t) for r, t in zip(responses, targets))

    metrics = {
        "exact_match": sum(_normalize(r) == _normalize(t) for r, t in zip(responses, targets)) / len(docs),
        "relaxed_accuracy": correct / len(docs),
    }
    print(f"chartqa: {metrics} ({elapsed:.2f}s)")

    return {
        "task": "chartqa",
        "metrics": metrics,
        "num_samples": len(docs),
        "time_seconds": round(elapsed, 2),
    }
