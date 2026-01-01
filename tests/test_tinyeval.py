"""Tests for tinyeval: HTTP client, image handling, metrics, task functions."""

import asyncio
import base64
from unittest.mock import AsyncMock, patch

import pytest


class TestImageHandling:
    """Image encoding and multimodal message building."""

    def test_encode_pil_image_and_string_passthrough(self):
        """PIL images encode to base64; strings pass through unchanged."""
        from tinyeval import _encode_image

        try:
            from PIL import Image

            img = Image.new("RGB", (10, 10), color="red")
            b64 = _encode_image(img)
            assert base64.b64decode(b64)  # Valid base64
        except ImportError:
            pytest.skip("PIL not available")

        # Strings pass through
        url = "https://example.com/image.png"
        assert _encode_image(url) == url

    def test_vision_message_structure(self):
        """Vision API message: images first, text second, <image> placeholder removed."""
        from tinyeval import _build_vision_message

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
        from tinyeval import _normalize

        assert _normalize("Hello") == "hello"
        assert _normalize("$42") == "42"
        assert _normalize("1,234") == "1234"
        assert _normalize("answer.") == "answer"

    def test_relaxed_match_exact(self):
        """Exact match cases."""
        from tinyeval import _relaxed_match

        assert _relaxed_match("42", "42") == 1.0
        assert _relaxed_match("Yes", "yes") == 1.0
        assert _relaxed_match("FINAL ANSWER: 42", "42") == 1.0

    def test_relaxed_match_numeric_tolerance(self):
        """5% numeric tolerance."""
        from tinyeval import _relaxed_match

        assert _relaxed_match("42", "40") == 1.0  # Within 5%
        assert _relaxed_match("50", "40") == 0.0  # >5%
        assert _relaxed_match("0", "0") == 1.0  # Zero case


class TestGSM8K:
    """GSM8K task functions."""

    def test_format_prompt(self):
        """GSM8K prompt includes question and few-shot examples."""
        from tinyeval import _format_gsm8k_prompt

        prompt = _format_gsm8k_prompt("What is 2 + 2?")
        assert "What is 2 + 2?" in prompt
        assert "The final answer is" in prompt
        assert "15 trees" in prompt  # First few-shot example

    def test_extract_answer(self):
        """Extract numeric answer from response."""
        from tinyeval import _extract_gsm8k_answer

        assert _extract_gsm8k_answer("The final answer is 42") == "42"
        assert _extract_gsm8k_answer("The final answer is $1,234") == "$1,234"
        assert _extract_gsm8k_answer("Some text with 42 in it") == "42"


class TestChartQA:
    """ChartQA task functions."""

    def test_format_prompt(self):
        """ChartQA prompt includes query and image placeholder."""
        from tinyeval import _format_chartqa_prompt

        prompt = _format_chartqa_prompt("What is the total revenue?")
        assert "What is the total revenue?" in prompt
        assert "FINAL ANSWER:" in prompt
        assert "<image>" in prompt


class TestHTTPClient:
    """HTTP client (complete function)."""

    def test_complete_text_prompts(self):
        """complete() handles text prompts and returns responses."""
        from tinyeval import APIConfig, complete

        config = APIConfig(url="http://test.com/v1/chat/completions", model="gpt-4")
        mock_response = {"choices": [{"message": {"content": "The answer is 42"}}]}

        class MockResp:
            ok = True

            async def json(self):
                return mock_response

        class MockContextManager:
            async def __aenter__(self):
                return MockResp()

            async def __aexit__(self, *args):
                pass

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager()
            )
            responses = asyncio.run(complete(["Test prompt"], config))

        assert responses[0] == "The answer is 42"

    def test_complete_multimodal_prompts(self):
        """complete() handles (text, images) tuples for multimodal."""
        from tinyeval import APIConfig, complete

        config = APIConfig(url="http://test.com/v1/chat/completions", model="gpt-4-vision")
        mock_response = {"choices": [{"message": {"content": "I see a chart"}}]}

        class MockResp:
            ok = True

            async def json(self):
                return mock_response

        class MockContextManager:
            async def __aenter__(self):
                return MockResp()

            async def __aexit__(self, *args):
                pass

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager()
            )
            # Multimodal prompt as tuple (text, images)
            responses = asyncio.run(complete([("Describe this chart", ["base64img"])], config))

        assert responses[0] == "I see a chart"


class TestTasks:
    """Task registry."""

    def test_tasks_registered(self):
        """Both tasks are registered."""
        from tinyeval import TASKS

        assert "gsm8k_llama" in TASKS
        assert "chartqa" in TASKS
        assert len(TASKS) == 2
