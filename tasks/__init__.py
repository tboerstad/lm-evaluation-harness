"""
Task registry for tinyeval.

Maps task names to Task class instances:
- gsm8k_llama: 8-shot grade school math
- chartqa: multimodal chart understanding

Architecture:
    TASKS is a dict[str, Task] mapping names to Task instances.
    Each Task implements the Task ABC from core.py.
"""

from core import Task
from tasks.chartqa import ChartQATask, _format_chartqa_prompt, _relaxed_match
from tasks.gsm8k import GSM8KTask, _extract_gsm8k_answer, _format_gsm8k_prompt

# Task instances - each Task knows its own name
_gsm8k_task = GSM8KTask()
_chartqa_task = ChartQATask()

# Registry mapping task names to Task instances
TASKS: dict[str, Task] = {
    _gsm8k_task.name: _gsm8k_task,
    _chartqa_task.name: _chartqa_task,
}

__all__ = [
    "TASKS",
    "GSM8KTask",
    "ChartQATask",
    # Helper functions exported for testing
    "_format_gsm8k_prompt",
    "_extract_gsm8k_answer",
    "_format_chartqa_prompt",
    "_relaxed_match",
]
