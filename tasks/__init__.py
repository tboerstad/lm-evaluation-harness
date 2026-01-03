"""
Task registry for nano-eval.

Maps task names to Task instances:
- gsm8k_llama: 8-shot grade school math
- chartqa: multimodal chart understanding
"""

from tasks.chartqa import chartqa
from tasks.gsm8k import gsm8k_llama

TASKS = {"gsm8k_llama": gsm8k_llama, "chartqa": chartqa}

__all__ = ["TASKS", "gsm8k_llama", "chartqa"]
