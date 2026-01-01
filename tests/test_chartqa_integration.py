"""Integration tests for ChartQA: dataset loading, config parsing, instance building, and evaluation pipeline."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Tasks directory is now at root level
TASKS_DIR = Path(__file__).parent.parent / "tasks"


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


class TestChartQATaskConfig:
    """ChartQA YAML config loading."""

    @pytest.fixture
    def config_path(self):
        return TASKS_DIR / "chartqa" / "chartqa.yaml"

    def test_config_loads_with_multimodal_settings(self, config_path):
        """Config loads, is multimodal, has doc_to_text template and metrics."""
        from liteeval import TaskConfig
        config = TaskConfig.from_yaml(config_path)

        assert config.task == "chartqa"
        assert config.dataset_path == "HuggingFaceM4/ChartQA"
        assert config.is_multimodal
        assert "image" in config.doc_to_image
        assert "<image>" in config.doc_to_text and "{{query}}" in config.doc_to_text
        assert config.metric_list


class TestChartQAInstanceBuilding:
    """Instance building from ChartQA data."""

    @pytest.fixture(scope="class")
    def chartqa_samples(self):
        import datasets
        dataset = datasets.load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
        return list(dataset.take(3))

    @pytest.fixture
    def config(self):
        from liteeval import TaskConfig
        return TaskConfig.from_yaml(TASKS_DIR / "chartqa" / "chartqa.yaml")

    def test_instances_have_images_prompts_targets(self, chartqa_samples, config):
        """Instances contain images, prompts with query, and targets from labels."""
        from liteeval import build_instances
        instances = build_instances(config, chartqa_samples)

        assert len(instances) == len(chartqa_samples)
        for i, inst in enumerate(instances):
            assert inst.images, "Instance should have images"
            assert chartqa_samples[i]["query"] in inst.prompt
            assert inst.target == chartqa_samples[i]["label"][0]


class TestChartQAMultimodal:
    """Multimodal message building with ChartQA images."""

    @pytest.fixture(scope="class")
    def sample(self):
        import datasets
        dataset = datasets.load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
        return next(iter(dataset))

    def test_image_encodes_and_message_builds(self, sample):
        """Image encodes to base64, multimodal message has correct vision API format."""
        from liteeval import build_multimodal_message, encode_image_to_base64

        b64 = encode_image_to_base64(sample["image"])
        assert b64 and isinstance(b64, str)

        messages = build_multimodal_message(f"<image>{sample['query']}\nFinal Answer:", [sample["image"]])
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64," in content[0]["image_url"]["url"]
        assert content[1]["type"] == "text"
        assert "<image>" not in content[1]["text"]


class TestChartQAEndToEnd:
    """End-to-end pipeline with mocked API."""

    def test_full_pipeline_with_mock(self):
        """Build instances, run generation with mock API, compute metrics."""
        import datasets
        from liteeval import APIConfig, TaskConfig, build_instances, compute_metrics, run_generation

        samples = list(datasets.load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True).take(2))
        config = TaskConfig.from_yaml(TASKS_DIR / "chartqa" / "chartqa.yaml")
        instances = build_instances(config, samples)

        api_config = APIConfig(base_url="http://mock/v1/chat/completions", model="mock", api_key="test")
        mock_response = {"choices": [{"message": {"content": f"Final Answer: {instances[0].target}"}}]}

        class MockResp:
            ok = True
            async def json(self):
                return mock_response

        class MockContextManager:
            async def __aenter__(self):
                return MockResp()
            async def __aexit__(self, *args):
                pass

        with patch("liteeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(post=lambda *a, **k: MockContextManager())
            instances = asyncio.run(run_generation(instances, api_config, is_multimodal=True))

        assert all(inst.response for inst in instances)
        metrics = compute_metrics(instances, config)
        assert "relaxed_accuracy" in metrics or "exact_match" in metrics
