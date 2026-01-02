"""Type definitions for tinyeval."""

from __future__ import annotations

from typing import TypedDict


class Metrics(TypedDict):
    exact_match: float
    relaxed_accuracy: float


class TaskResult(TypedDict):
    task: str
    metrics: Metrics
    num_samples: int
    elapsed: float


class RunConfig(TypedDict):
    model: str
    max_samples: int | None


class EvalOutput(TypedDict):
    results: dict[str, TaskResult]
    total_seconds: float
    config: RunConfig
