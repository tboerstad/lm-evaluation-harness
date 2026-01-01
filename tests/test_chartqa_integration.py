"""Integration tests for ChartQA: dataset loading and multimodal handling."""

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
