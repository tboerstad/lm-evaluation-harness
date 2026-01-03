"""
End-to-end tests for tinyeval CLI.

Tests gsm8k_llama and chartqa tasks with real datasets and mocked API responses.
"""

from unittest.mock import patch

import httpx
import pytest
import respx

from core import APIConfig, Task, compute_task_hash
from tasks import TASKS
from tasks.chartqa import score as chartqa_score
from tasks.gsm8k import score as gsm8k_score
from tinyeval import evaluate

RESPONSES = [
    "The final answer is 1",
    "The final answer is 2",
    "The final answer is 3",
    "The final answer is 4",
    "The final answer is 5",
    "The final answer is 6",
    "The final answer is 7",
    "The final answer is 8",
    "The final answer is 9",
    "The final answer is 10",
]


class TestE2E:
    """End-to-end tests with real datasets and mocked API responses."""

    @pytest.mark.asyncio
    async def test_gsm8k_evaluation(self):
        """GSM8K task produces deterministic hash."""
        call_count = 0

        def mock_response(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            response = RESPONSES[call_count % len(RESPONSES)]
            call_count += 1
            return httpx.Response(
                200, json={"choices": [{"message": {"content": response}}]}
            )

        real_samples = TASKS["gsm8k_llama"].samples(10)
        expected_hash = compute_task_hash(real_samples)
        mock_task = Task(
            name="gsm8k_llama",
            samples=lambda max_samples: real_samples[:max_samples]
            if max_samples
            else real_samples,
            score=gsm8k_score,
        )

        with patch.dict("tinyeval.TASKS", {"gsm8k_llama": mock_task}, clear=True):
            with respx.mock as mock:
                mock.post("http://test.com/v1").mock(side_effect=mock_response)
                config = APIConfig(
                    url="http://test.com/v1", model="test-model", seed=42
                )
                result = await evaluate(["gsm8k_llama"], config, max_samples=10)

        assert result["results"]["gsm8k_llama"]["task_hash"] == expected_hash
        assert compute_task_hash(real_samples) == expected_hash
        assert result["results"]["gsm8k_llama"]["num_samples"] == 10

    @pytest.mark.asyncio
    async def test_chartqa_evaluation(self):
        """ChartQA task sends image data and produces deterministic hash."""
        call_count = 0
        received_image_data = False

        def mock_response(request: httpx.Request) -> httpx.Response:
            nonlocal call_count, received_image_data
            body = request.content.decode()
            if "data:image/png;base64," in body:
                received_image_data = True
            response = RESPONSES[call_count % len(RESPONSES)]
            call_count += 1
            return httpx.Response(
                200, json={"choices": [{"message": {"content": response}}]}
            )

        real_samples = TASKS["chartqa"].samples(10)
        expected_hash = compute_task_hash(real_samples)
        mock_task = Task(
            name="chartqa",
            samples=lambda max_samples: real_samples[:max_samples]
            if max_samples
            else real_samples,
            score=chartqa_score,
        )

        with patch.dict("tinyeval.TASKS", {"chartqa": mock_task}, clear=True):
            with respx.mock as mock:
                mock.post("http://test.com/v1").mock(side_effect=mock_response)
                config = APIConfig(
                    url="http://test.com/v1", model="test-model", seed=42
                )
                result = await evaluate(["chartqa"], config, max_samples=10)

        assert received_image_data, "Expected image data in request payload"
        assert result["results"]["chartqa"]["task_hash"] == expected_hash
        assert compute_task_hash(real_samples) == expected_hash
        assert result["results"]["chartqa"]["num_samples"] == 10
