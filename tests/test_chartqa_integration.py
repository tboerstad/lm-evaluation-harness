"""Integration tests for ChartQA: dataset loading and evaluation pipeline."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


class TestChartQADataset:
    """ChartQA dataset structure validation."""

    @pytest.fixture(scope="class")
    def chartqa_samples(self):
        """Load ChartQA samples (streaming for speed)."""
        import datasets

        dataset = datasets.load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
        return list(dataset.take(3))

    def test_dataset_structure(self, chartqa_samples):
        """Dataset has image (PIL), query (str), label (list) fields."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        assert len(chartqa_samples) == 3
        sample = chartqa_samples[0]
        assert isinstance(sample["image"], Image.Image)
        assert isinstance(sample["query"], str) and sample["query"]
        assert isinstance(sample["label"], list) and sample["label"]


class TestChartQATaskFunction:
    """ChartQA task function validation."""

    def test_task_is_registered(self):
        """ChartQA is registered in TASKS."""
        from tinyeval import TASKS

        assert "chartqa" in TASKS
        assert callable(TASKS["chartqa"])

    def test_prompt_formatting(self):
        """Prompt includes query and FINAL ANSWER instruction."""
        from tinyeval import _format_chartqa_prompt

        prompt = _format_chartqa_prompt("What is the total?")
        assert "What is the total?" in prompt
        assert "FINAL ANSWER:" in prompt
        assert "<image>" in prompt


class TestChartQAMultimodal:
    """Multimodal message building with ChartQA images."""

    @pytest.fixture(scope="class")
    def sample(self):
        import datasets

        dataset = datasets.load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
        return next(iter(dataset))

    def test_image_encodes_and_message_builds(self, sample):
        """Image encodes to base64, multimodal message has correct vision API format."""
        from tinyeval import _build_vision_message, _encode_image

        b64 = _encode_image(sample["image"])
        assert b64 and isinstance(b64, str)

        messages = _build_vision_message(f"<image>{sample['query']}\nFINAL ANSWER:", [sample["image"]])
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64," in content[0]["image_url"]["url"]
        assert content[1]["type"] == "text"
        assert "<image>" not in content[1]["text"]


class TestChartQAEndToEnd:
    """End-to-end pipeline with mocked API."""

    def test_complete_with_multimodal_prompts(self):
        """complete() handles multimodal prompts with images."""
        from tinyeval import APIConfig, _format_chartqa_prompt, complete

        config = APIConfig(url="http://mock/v1/chat/completions", model="mock", api_key="test")

        # Create multimodal prompt
        prompts = [(_format_chartqa_prompt("What is the value?"), ["base64_image_data"])]

        mock_response = {"choices": [{"message": {"content": "FINAL ANSWER: 42"}}]}

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
            responses = asyncio.run(complete(prompts, config))

        assert len(responses) == 1
        assert "42" in responses[0]

    def test_relaxed_match_scoring(self):
        """Relaxed match handles FINAL ANSWER extraction and numeric tolerance."""
        from tinyeval import _relaxed_match

        # Exact match via FINAL ANSWER
        assert _relaxed_match("FINAL ANSWER: 42", "42") == 1.0
        # Numeric tolerance (5%)
        assert _relaxed_match("FINAL ANSWER: 42", "40") == 1.0
        # Case insensitive
        assert _relaxed_match("FINAL ANSWER: Yes", "yes") == 1.0
        # Miss
        assert _relaxed_match("FINAL ANSWER: 100", "42") == 0.0
