"""
End-to-end tests for nano-eval CLI.

Tests use real datasets with mocked API responses via respx.
Samples are loaded before mocking to avoid respx/proxy conflicts.
"""

import json
import sys
from unittest.mock import patch

import respx
from httpx import Response

from core import Task
from nano_eval import main
from tasks.chartqa import samples as load_chartqa_samples, score as chartqa_score
from tasks.gsm8k import samples as load_gsm8k_samples, score as gsm8k_score

# GSM8K: Single correct response used for all requests (order-independent)
# All samples get the same response - we just verify the pipeline works
GSM8K_RESPONSE = "The final answer is 18"

# ChartQA: Single correct response used for all requests (order-independent)
CHARTQA_RESPONSE = "FINAL ANSWER: 14"

GSM8K_HASH = "f97c1e3ca96e715651955a910fb60a3a05528cd0de8b68ff3b43194ef05cf953"
CHARTQA_HASH = "c6043b6621aa01df9276f665b2a8a1be6ac28c86caf5822ae3b4333a4b793caf"


class TestE2E:
    """End-to-end tests with real datasets and mocked API responses."""

    def test_gsm8k_evaluation(self, tmp_path):
        """GSM8K evaluation with real dataset, mocked API."""
        real_samples = load_gsm8k_samples(10)

        task = Task(
            name="gsm8k_llama", samples=lambda n: real_samples, score=gsm8k_score
        )

        with respx.mock:
            respx.post("http://test.com/v1/chat/completions").mock(
                return_value=Response(
                    200, json={"choices": [{"message": {"content": GSM8K_RESPONSE}}]}
                )
            )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "nano-eval",
                        "--tasks",
                        "gsm8k_llama",
                        "--model_args",
                        "model=test,base_url=http://test.com/v1",
                        "--max_samples",
                        "10",
                        "--output_path",
                        str(tmp_path),
                        "--log_samples",
                    ],
                ),
                patch.dict("tasks.TASKS", {"gsm8k_llama": task}),
            ):
                main()

        results = json.loads((tmp_path / "results.json").read_text())
        assert results["results"]["gsm8k_llama"]["metrics"]["exact_match"] == 0.1
        assert results["results"]["gsm8k_llama"]["task_hash"] == GSM8K_HASH

        samples = [
            json.loads(line)
            for line in (tmp_path / "samples_gsm8k_llama.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(samples) == 10
        assert samples[0]["target"] == "18"
        assert samples[0]["response"] == GSM8K_RESPONSE
        assert samples[0]["exact_match"] == 1.0

    def test_chartqa_evaluation(self, tmp_path):
        """ChartQA evaluation with real dataset, mocked API."""
        real_samples = load_chartqa_samples(10)

        task = Task(name="chartqa", samples=lambda n: real_samples, score=chartqa_score)

        with respx.mock:
            respx.post("http://test.com/v1/chat/completions").mock(
                return_value=Response(
                    200, json={"choices": [{"message": {"content": CHARTQA_RESPONSE}}]}
                )
            )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "nano-eval",
                        "--tasks",
                        "chartqa",
                        "--model_args",
                        "model=test,base_url=http://test.com/v1",
                        "--max_samples",
                        "10",
                        "--output_path",
                        str(tmp_path),
                        "--log_samples",
                    ],
                ),
                patch.dict("tasks.TASKS", {"chartqa": task}),
            ):
                main()

        results = json.loads((tmp_path / "results.json").read_text())
        assert results["results"]["chartqa"]["metrics"]["exact_match"] == 0.1
        assert results["results"]["chartqa"]["task_hash"] == CHARTQA_HASH

        samples = [
            json.loads(line)
            for line in (tmp_path / "samples_chartqa.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(samples) == 10
        assert samples[0]["target"] == "14"
        assert samples[0]["response"] == CHARTQA_RESPONSE
        assert samples[0]["exact_match"] == 1.0
