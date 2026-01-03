"""
End-to-end tests for tinyeval CLI.

Tests use real datasets with mocked API responses via respx.
Samples are loaded before mocking to avoid respx/proxy conflicts.
"""

import json
import sys
from unittest.mock import patch

import respx
from httpx import Response

from core import Task
from tasks.chartqa import samples as load_chartqa_samples, score as chartqa_score
from tasks.gsm8k import samples as load_gsm8k_samples, score as gsm8k_score
from tinyeval import main

# GSM8K: 10 mock responses (7 correct, 3 wrong = 70% accuracy)
# Target answers: 18, 3, 70000, 540, 20, 64, 260, 160, 45, 460
GSM8K_RESPONSES = [
    "The final answer is 18",
    "The final answer is 3",
    "The final answer is 70000",
    "The final answer is 999",
    "The final answer is 20",
    "The final answer is 64",
    "The final answer is 260",
    "The final answer is 999",
    "The final answer is 999",
    "The final answer is 460",
]

# ChartQA: 10 mock responses (7 correct, 3 wrong = 70% accuracy)
# Target answers: 14, 0.57, 3, No, 23, 6, 62, Yes, Inspired, 0.03
CHARTQA_RESPONSES = [
    "FINAL ANSWER: 14",
    "FINAL ANSWER: 0.57",
    "FINAL ANSWER: 3",
    "FINAL ANSWER: No",
    "FINAL ANSWER: 999",
    "FINAL ANSWER: 6",
    "FINAL ANSWER: 62",
    "FINAL ANSWER: wrong",
    "FINAL ANSWER: wrong",
    "FINAL ANSWER: 0.03",
]

GSM8K_HASH = "f97c1e3ca96e715651955a910fb60a3a05528cd0de8b68ff3b43194ef05cf953"
CHARTQA_HASH = "c6043b6621aa01df9276f665b2a8a1be6ac28c86caf5822ae3b4333a4b793caf"


class TestE2E:
    """End-to-end tests with real datasets and mocked API responses."""

    def test_gsm8k_evaluation(self, tmp_path):
        """GSM8K evaluation with real dataset, mocked API."""
        real_samples = load_gsm8k_samples(10)
        call_count = 0

        def api_response(request):
            nonlocal call_count
            content = GSM8K_RESPONSES[call_count % len(GSM8K_RESPONSES)]
            call_count += 1
            return Response(200, json={"choices": [{"message": {"content": content}}]})

        task = Task(
            name="gsm8k_llama", samples=lambda n: real_samples, score=gsm8k_score
        )

        with respx.mock:
            respx.post("http://test.com/v1").mock(side_effect=api_response)

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "tinyeval",
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
        assert results["results"]["gsm8k_llama"]["metrics"]["exact_match"] == 0.7
        assert results["results"]["gsm8k_llama"]["task_hash"] == GSM8K_HASH

        samples = [
            json.loads(line)
            for line in (tmp_path / "samples_gsm8k_llama.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(samples) == 10
        assert samples[0]["doc_id"] == 0
        assert samples[0]["target"] == "18"
        assert samples[0]["response"] == "The final answer is 18"
        assert samples[0]["exact_match"] == 1.0
        assert samples[3]["target"] == "540"
        assert samples[3]["exact_match"] == 0.0

    def test_chartqa_evaluation(self, tmp_path):
        """ChartQA evaluation with real dataset, mocked API."""
        real_samples = load_chartqa_samples(10)
        call_count = 0

        def api_response(request):
            nonlocal call_count
            content = CHARTQA_RESPONSES[call_count % len(CHARTQA_RESPONSES)]
            call_count += 1
            return Response(200, json={"choices": [{"message": {"content": content}}]})

        task = Task(name="chartqa", samples=lambda n: real_samples, score=chartqa_score)

        with respx.mock:
            respx.post("http://test.com/v1").mock(side_effect=api_response)

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "tinyeval",
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
        assert results["results"]["chartqa"]["metrics"]["exact_match"] == 0.7
        assert results["results"]["chartqa"]["task_hash"] == CHARTQA_HASH

        samples = [
            json.loads(line)
            for line in (tmp_path / "samples_chartqa.jsonl")
            .read_text()
            .strip()
            .split("\n")
        ]
        assert len(samples) == 10
        assert samples[0]["doc_id"] == 0
        assert samples[0]["target"] == "14"
        assert samples[0]["response"] == "FINAL ANSWER: 14"
        assert samples[0]["exact_match"] == 1.0
        assert samples[4]["target"] == "23"
        assert samples[4]["exact_match"] == 0.0
