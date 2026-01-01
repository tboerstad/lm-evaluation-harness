"""Task definitions for tinyeval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gsm8k_llama import get_gsm8k_llama_config
from .chartqa import get_chartqa_config


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
    "gsm8k_llama": get_gsm8k_llama_config(TaskConfig),
    "chartqa": get_chartqa_config(TaskConfig),
}
