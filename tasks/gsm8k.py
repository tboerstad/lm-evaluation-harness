"""
GSM8K evaluation - grade school math with chain-of-thought.

Responsibilities:
- Load gsm8k dataset (test → train, stops at limit)
- Format 8-shot prompts with reasoning examples
- Extract numeric answers from responses
- Compute exact_match (normalized string comparison)
"""

from __future__ import annotations

import logging
import re
import time

import datasets

from core import CompletionService, Task, TaskResult, _normalize

logger = logging.getLogger(__name__)

GSM8K_FEWSHOT = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8",
    ),
]

GSM8K_STOP = [
    "<|eot_id|>",
    "<|start_header_id|>user<|end_header_id|>",
    "Q:",
    "</s>",
    "<|im_end|>",
]


_GSM8K_TEMPLATE = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
)


def _format_gsm8k_prompt(question: str) -> str:
    """Format GSM8K prompt with few-shot examples."""
    parts = [_GSM8K_TEMPLATE.format(question=q) + f"\n {a}" for q, a in GSM8K_FEWSHOT]
    parts.append(_GSM8K_TEMPLATE.format(question=question))
    return "\n\n".join(parts)


_NUM_PATTERN = r"-?[$0-9.,]{2,}|-?[0-9]+"


def _parse_target(answer: str) -> str:
    """Parse target answer from GSM8K format, handling missing #### delimiter."""
    parts = answer.split("####")
    if len(parts) < 2:
        return answer.strip()
    return parts[-1].strip()


def _extract_gsm8k_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response."""
    # Check for explicit template first
    if match := re.search(rf"The final answer is ({_NUM_PATTERN})", response):
        return match.group(1)

    # Fallback: Find ALL numbers and return the LAST one
    matches = re.findall(rf"({_NUM_PATTERN})", response)
    if matches:
        return matches[-1]

    return response


class GSM8KTask(Task):
    """
    GSM8K evaluation task - grade school math with chain of thought.

    This task implements the Task protocol, receiving a CompletionService
    via dependency injection rather than directly calling API functions.
    """

    @property
    def name(self) -> str:
        return "gsm8k_llama"

    async def evaluate(
        self,
        completion_service: CompletionService,
        max_samples: int | None = None,
    ) -> TaskResult:
        """
        Evaluate GSM8K - grade school math with chain of thought.

        Returns TaskResult with exact_match, num_samples, elapsed.
        """
        docs = self._load_dataset(max_samples)
        targets = [_parse_target(d["answer"]) for d in docs]
        prompts = [_format_gsm8k_prompt(d["question"]) for d in docs]

        logger.info("Evaluating: %s (%d samples)", self.name, len(docs))
        t0 = time.perf_counter()
        responses = await completion_service.complete(prompts, stop=GSM8K_STOP)
        elapsed = time.perf_counter() - t0

        correct = sum(
            _normalize(_extract_gsm8k_answer(r)) == _normalize(t)
            for r, t in zip(responses, targets)
        )

        metrics = {
            "exact_match": correct / len(docs),
            "relaxed_accuracy": correct / len(docs),
        }
        logger.info("%s: %s (%.2fs)", self.name, metrics, elapsed)

        return {
            "task": self.name,
            "metrics": metrics,
            "num_samples": len(docs),
            "elapsed": round(elapsed, 2),
        }

    def _load_dataset(self, max_samples: int | None) -> list[dict]:
        """Load GSM8K dataset samples."""
        docs = []
        for split in ["test", "train"]:
            ds = datasets.load_dataset("gsm8k", "main", split=split, streaming=True)
            for doc in ds:
                docs.append(doc)
                if max_samples and len(docs) >= max_samples:
                    break
            if max_samples and len(docs) >= max_samples:
                break
        return docs
