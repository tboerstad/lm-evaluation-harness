"""ChartQA task - Multimodal chart question answering."""
from __future__ import annotations

from typing import Any


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


def get_chartqa_config(TaskConfig):
    """Return the ChartQA task configuration."""
    return TaskConfig(
        task="chartqa",
        dataset_path="HuggingFaceM4/ChartQA",
        doc_to_text=chartqa_doc_to_text,
        doc_to_target=chartqa_doc_to_target,
        doc_to_image=["image"],
        max_tokens=512,
    )
