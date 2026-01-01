"""
Task registry for tinyeval.

Maps task names to evaluation functions:
- gsm8k_llama: 8-shot grade school math
- chartqa: multimodal chart understanding
"""

from tasks.chartqa import eval_chartqa
from tasks.gsm8k import eval_gsm8k

TASKS = {"gsm8k_llama": eval_gsm8k, "chartqa": eval_chartqa}

__all__ = ["TASKS", "eval_gsm8k", "eval_chartqa"]
