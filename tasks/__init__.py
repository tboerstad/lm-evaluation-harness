"""Task registry for tinyeval.

TASKS maps task names to task modules. Each module provides:
- NAME: str - task identifier
- STOP: list[str] | None - stop sequences
- load(limit) -> list - load dataset
- prompt(doc) -> str | tuple[str, list] - format prompt
- score(responses, docs) -> dict - compute metrics
"""

from tasks import chartqa, gsm8k

# Map task names to modules
TASKS = {
    gsm8k.NAME: gsm8k,
    chartqa.NAME: chartqa,
}

__all__ = ["TASKS", "gsm8k", "chartqa"]
