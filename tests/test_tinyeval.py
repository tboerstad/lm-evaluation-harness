"""
End-to-end tests for tinyeval CLI.

Tests gsm8k_llama and chartqa tasks with mocked datasets and API responses.
"""

from unittest.mock import patch

import httpx
import pytest
import respx
from PIL import Image

from core import APIConfig, Sample, Task, compute_task_hash
from tasks.chartqa import score as chartqa_score
from tasks.gsm8k import score as gsm8k_score
from tinyeval import evaluate

GSM8K_QUESTIONS = [
    "Tom has 5 apples and buys 3 more. How many apples does Tom have?",
    "Sara has 12 cookies and gives 4 to her friend. How many cookies left?",
    "A store sells 7 books Monday and 8 Tuesday. How many total?",
    "Mike ran 3 miles each day for 4 days. How many miles total?",
    "Lisa has 20 stickers to share among 5 friends. How many each?",
    "9 birds on a tree and 6 fly away. How many left?",
    "A baker makes 24 cupcakes and puts 6 in each box. How many boxes?",
    "John has $15 and spends $7 on a toy. How much left?",
    "A garden has 4 rows with 5 flowers each. How many flowers total?",
    "Emma collects 8 seashells Saturday and 9 Sunday. How many total?",
]
GSM8K_TARGETS = ["8", "8", "15", "12", "4", "3", "4", "8", "20", "17"]
GSM8K_RESPONSES = [
    "5 + 3 = 8. The final answer is 8",
    "12 - 4 = 8. The final answer is 8",
    "7 + 8 = 15. The final answer is 15",
    "3 * 4 = 12. The final answer is 12",
    "20 / 5 = 4. The final answer is 4",
    "9 - 6 = 3. The final answer is 3",
    "24 / 6 = 4. The final answer is 4",
    "Wrong. The final answer is 99",
    "Wrong. The final answer is 99",
    "Wrong. The final answer is 99",
]

CHARTQA_QUESTIONS = [
    "What is the value of the blue bar?",
    "What percentage does the red section represent?",
    "How many categories are shown?",
    "What is the difference between highest and lowest?",
    "What is the total sum of all bars?",
    "Which category has the smallest value?",
    "What is the average value?",
    "What percentage does the green bar represent?",
    "How many bars exceed 50?",
    "What is the median value?",
]
CHARTQA_TARGETS = ["75", "25", "5", "40", "200", "Electronics", "40", "15", "3", "35"]
CHARTQA_RESPONSES = [
    "FINAL ANSWER: 75",
    "FINAL ANSWER: 25",
    "FINAL ANSWER: 5",
    "FINAL ANSWER: 40",
    "FINAL ANSWER: 200",
    "FINAL ANSWER: Electronics",
    "FINAL ANSWER: 40",
    "FINAL ANSWER: 99",
    "FINAL ANSWER: 99",
    "FINAL ANSWER: 99",
]


def _gsm8k_samples(max_samples: int | None = None) -> list[Sample]:
    samples = [
        Sample(prompt=q, target=t) for q, t in zip(GSM8K_QUESTIONS, GSM8K_TARGETS)
    ]
    return samples[:max_samples] if max_samples else samples


def _chartqa_samples(max_samples: int | None = None) -> list[Sample]:
    img = Image.new("RGB", (100, 100), color="white")
    samples = [
        Sample(prompt=(q, [img]), target=t)
        for q, t in zip(CHARTQA_QUESTIONS, CHARTQA_TARGETS)
    ]
    return samples[:max_samples] if max_samples else samples


class TestE2E:
    """End-to-end tests with mocked datasets and API responses."""

    @pytest.mark.asyncio
    async def test_gsm8k_evaluation(self):
        """GSM8K task produces deterministic hash and correct metrics."""
        call_count = 0

        def mock_response(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            response = GSM8K_RESPONSES[call_count % len(GSM8K_RESPONSES)]
            call_count += 1
            return httpx.Response(
                200, json={"choices": [{"message": {"content": response}}]}
            )

        mock_task = Task(name="gsm8k_llama", samples=_gsm8k_samples, score=gsm8k_score)
        samples = _gsm8k_samples(10)
        expected_hash = compute_task_hash(samples)

        with patch.dict("tinyeval.TASKS", {"gsm8k_llama": mock_task}, clear=True):
            with respx.mock as mock:
                mock.post("http://test.com/v1").mock(side_effect=mock_response)
                config = APIConfig(
                    url="http://test.com/v1", model="test-model", seed=42
                )
                result = await evaluate(["gsm8k_llama"], config, max_samples=10)

        assert result["results"]["gsm8k_llama"]["task_hash"] == expected_hash
        assert compute_task_hash(samples) == expected_hash
        assert result["results"]["gsm8k_llama"]["metrics"]["exact_match"] == 0.7
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
            response = CHARTQA_RESPONSES[call_count % len(CHARTQA_RESPONSES)]
            call_count += 1
            return httpx.Response(
                200, json={"choices": [{"message": {"content": response}}]}
            )

        mock_task = Task(name="chartqa", samples=_chartqa_samples, score=chartqa_score)
        samples = _chartqa_samples(10)
        expected_hash = compute_task_hash(samples)

        with patch.dict("tinyeval.TASKS", {"chartqa": mock_task}, clear=True):
            with respx.mock as mock:
                mock.post("http://test.com/v1").mock(side_effect=mock_response)
                config = APIConfig(
                    url="http://test.com/v1", model="test-model", seed=42
                )
                result = await evaluate(["chartqa"], config, max_samples=10)

        assert received_image_data, "Expected image data in request payload"
        assert result["results"]["chartqa"]["task_hash"] == expected_hash
        assert compute_task_hash(samples) == expected_hash
        assert result["results"]["chartqa"]["metrics"]["exact_match"] == 0.7
        assert result["results"]["chartqa"]["num_samples"] == 10
