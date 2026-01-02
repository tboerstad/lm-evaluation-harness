"""
Test suite for tinyeval.

Coverage:
- Image encoding (PIL→base64, paths, URL rejection)
- Text normalization and metrics
- GSM8K: prompt formatting, answer extraction
- ChartQA: prompt formatting, relaxed matching
- HTTP client: text and multimodal completions
- Task classes: Task ABC implementation
- Integration: end-to-end pipeline with dependency injection
"""

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from core import (
    APIConfig,
    DefaultCompletionService,
    Task,
    _build_vision_message,
    _encode_image,
    _normalize,
    complete,
)
from tasks import TASKS, ChartQATask, GSM8KTask
from tasks.chartqa import _format_chartqa_prompt, _relaxed_match
from tasks.gsm8k import _extract_gsm8k_answer, _format_gsm8k_prompt
from tinyeval import evaluate


def _make_mock_session(response_data: dict):
    """Create a mock aiohttp session that returns the given response."""

    class MockResp:
        ok = True

        async def json(self):
            return response_data

    class MockContextManager:
        async def __aenter__(self):
            return MockResp()

        async def __aexit__(self, *args):
            pass

    mock = AsyncMock(post=lambda *a, **k: MockContextManager())
    return mock


class MockCompletionService:
    """Mock CompletionService for testing tasks in isolation."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.prompts_received: list[Any] = []

    async def complete(
        self,
        prompts: list[str | tuple[str, list[Any]]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Return pre-configured responses."""
        self.prompts_received = list(prompts)
        return self._responses[: len(prompts)]


class TestImageHandling:
    """Image encoding and multimodal message building."""

    def test_encode_pil_image_and_string_passthrough(self):
        """PIL images encode to base64; local strings pass through; URLs rejected."""
        img = Image.new("RGB", (10, 10), color="red")
        b64 = _encode_image(img)
        assert base64.b64decode(b64)

        # Local string paths pass through
        local_path = "/path/to/image.png"
        assert _encode_image(local_path) == local_path

        # Remote URLs are rejected
        with pytest.raises(AssertionError, match="Remote image URLs are not supported"):
            _encode_image("https://example.com/image.png")
        with pytest.raises(AssertionError, match="Remote image URLs are not supported"):
            _encode_image("http://example.com/image.png")

    def test_vision_message_structure(self):
        """Vision API message: images first, text second, <image> placeholder removed."""
        messages = _build_vision_message("What is in <image>this chart?", ["abc123"])

        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in content[0]["image_url"]["url"]
        assert content[1]["type"] == "text"
        assert "<image>" not in content[1]["text"]


class TestMetrics:
    """Scoring functions."""

    def test_normalize(self):
        """Text normalization for comparison."""
        assert _normalize("Hello") == "hello"
        assert _normalize("$42") == "42"
        assert _normalize("1,234") == "1234"
        assert _normalize("answer.") == "answer"

    def test_relaxed_match_exact(self):
        """Exact match cases."""
        assert _relaxed_match("42", "42") == 1.0
        assert _relaxed_match("Yes", "yes") == 1.0
        assert _relaxed_match("FINAL ANSWER: 42", "42") == 1.0

    def test_relaxed_match_numeric_tolerance(self):
        """5% numeric tolerance."""
        assert _relaxed_match("42", "40") == 1.0  # Within 5%
        assert _relaxed_match("50", "40") == 0.0  # >5%
        assert _relaxed_match("0", "0") == 1.0  # Zero case


class TestGSM8K:
    """GSM8K task functions."""

    def test_format_prompt(self):
        """GSM8K prompt includes question and few-shot examples."""
        prompt = _format_gsm8k_prompt("What is 2 + 2?")
        assert "What is 2 + 2?" in prompt
        assert "The final answer is" in prompt
        assert "15 trees" in prompt  # First few-shot example

    def test_extract_answer(self):
        """Extract numeric answer from response."""
        assert _extract_gsm8k_answer("The final answer is 42") == "42"
        assert _extract_gsm8k_answer("The final answer is $1,234") == "$1,234"
        assert _extract_gsm8k_answer("Some text with 42 in it") == "42"


class TestChartQA:
    """ChartQA task functions."""

    def test_format_prompt(self):
        """ChartQA prompt includes query and image placeholder."""
        prompt = _format_chartqa_prompt("What is the total revenue?")
        assert "What is the total revenue?" in prompt
        assert "FINAL ANSWER:" in prompt
        assert "<image>" in prompt


class TestHTTPClient:
    """HTTP client (complete function)."""

    def test_complete_text_prompts(self):
        """complete() handles text prompts and returns responses."""
        config = APIConfig(url="http://test.com/v1/chat/completions", model="gpt-4")
        response = {"choices": [{"message": {"content": "The answer is 42"}}]}

        with patch("core.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = _make_mock_session(
                response
            )
            responses = asyncio.run(complete(["Test prompt"], config))

        assert responses[0] == "The answer is 42"

    def test_complete_multimodal_prompts(self):
        """complete() handles (text, images) tuples for multimodal."""
        config = APIConfig(
            url="http://test.com/v1/chat/completions", model="gpt-4-vision"
        )
        response = {"choices": [{"message": {"content": "I see a chart"}}]}

        with patch("core.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = _make_mock_session(
                response
            )
            responses = asyncio.run(
                complete([("Describe this chart", ["base64img"])], config)
            )

        assert responses[0] == "I see a chart"


class TestTaskClasses:
    """Task ABC implementation tests."""

    def test_gsm8k_task_is_task(self):
        """GSM8KTask is a proper Task implementation."""
        task = GSM8KTask()
        assert isinstance(task, Task)
        assert task.name == "gsm8k_llama"

    def test_chartqa_task_is_task(self):
        """ChartQATask is a proper Task implementation."""
        task = ChartQATask()
        assert isinstance(task, Task)
        assert task.name == "chartqa"

    def test_gsm8k_task_uses_completion_service(self):
        """GSM8KTask calls completion_service.complete() correctly."""
        task = GSM8KTask()
        mock_service = MockCompletionService(["The final answer is 4"])

        with patch.object(task, "_load_dataset") as mock_load:
            mock_load.return_value = [{"question": "2+2?", "answer": "#### 4"}]
            result = asyncio.run(task.evaluate(mock_service, max_samples=1))

        assert result["task"] == "gsm8k_llama"
        assert result["num_samples"] == 1
        assert len(mock_service.prompts_received) == 1

    def test_chartqa_task_uses_completion_service(self):
        """ChartQATask calls completion_service.complete() correctly."""
        task = ChartQATask()
        mock_service = MockCompletionService(["FINAL ANSWER: Yes"])

        with patch.object(task, "_load_dataset") as mock_load:
            mock_load.return_value = [
                {"query": "Is this a bar chart?", "label": "Yes", "image": "base64img"}
            ]
            result = asyncio.run(task.evaluate(mock_service, max_samples=1))

        assert result["task"] == "chartqa"
        assert result["num_samples"] == 1
        # Multimodal: prompt is (text, images) tuple
        assert isinstance(mock_service.prompts_received[0], tuple)


class TestDefaultCompletionService:
    """DefaultCompletionService implementation tests."""

    def test_implements_protocol(self):
        """DefaultCompletionService implements CompletionService protocol."""
        config = APIConfig(url="http://test.com", model="test")
        service = DefaultCompletionService(config)
        # Protocol check: has complete method with correct signature
        assert hasattr(service, "complete")
        assert callable(service.complete)


class TestTasks:
    """Task registry."""

    def test_tasks_registered(self):
        """Both tasks are registered."""
        assert "gsm8k_llama" in TASKS
        assert "chartqa" in TASKS
        assert len(TASKS) == 2

    def test_tasks_are_task_instances(self):
        """Registry contains Task instances."""
        for name, task in TASKS.items():
            assert isinstance(task, Task)
            assert task.name == name


class TestIntegration:
    """End-to-end integration tests."""

    def test_evaluate_gsm8k_end_to_end(self):
        """Full GSM8K evaluation pipeline with mocked API and dataset."""
        mock_docs = [{"question": "What is 2 + 2?", "answer": "#### 4"}]
        response = {"choices": [{"message": {"content": "The final answer is 4"}}]}

        with (
            patch("tasks.gsm8k.datasets.load_dataset") as mock_ds,
            patch("core.aiohttp.ClientSession") as mock_session,
        ):
            # Mock streaming dataset as iterable
            mock_ds.return_value.__iter__ = lambda self: iter(mock_docs)
            mock_session.return_value.__aenter__.return_value = _make_mock_session(
                response
            )

            config = APIConfig(url="http://test.com/v1/chat/completions", model="test")
            result = asyncio.run(evaluate(["gsm8k_llama"], config, max_samples=1))

        assert result["results"]["gsm8k_llama"]["metrics"]["exact_match"] == 1.0
        assert result["results"]["gsm8k_llama"]["num_samples"] == 1

    def test_evaluate_invalid_task(self):
        """Evaluation raises ValueError for unknown task."""
        config = APIConfig(url="http://test.com", model="test")
        with pytest.raises(ValueError, match="Unknown task"):
            asyncio.run(evaluate(["nonexistent_task"], config))
