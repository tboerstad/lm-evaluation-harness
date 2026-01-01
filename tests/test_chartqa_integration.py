"""Integration tests for ChartQA dataset download and evaluation.

This test verifies that:
1. ChartQA dataset can be downloaded from HuggingFace
2. Dataset has expected structure (image, query, label fields)
3. TaskConfig loads correctly from YAML
4. Instances can be built from ChartQA data
5. Multimodal message building works with real ChartQA images
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


class TestChartQADatasetDownload:
    """Tests for ChartQA dataset downloading and structure."""

    @pytest.fixture(scope="class")
    def chartqa_dataset(self):
        """Load a small sample of ChartQA dataset."""
        import datasets

        # Load the dataset with streaming to minimize download time
        dataset = datasets.load_dataset(
            "HuggingFaceM4/ChartQA",
            split="test",
            streaming=True,
        )
        # Take first 5 samples for testing
        samples = list(dataset.take(5))
        return samples

    def test_dataset_downloads_successfully(self, chartqa_dataset):
        """Test that ChartQA dataset can be downloaded."""
        assert len(chartqa_dataset) == 5
        assert len(chartqa_dataset) > 0

    def test_dataset_has_required_fields(self, chartqa_dataset):
        """Test that dataset has expected fields: image, query, label."""
        sample = chartqa_dataset[0]

        # ChartQA should have 'image', 'query', and 'label' fields
        assert "image" in sample, f"Missing 'image' field. Keys: {sample.keys()}"
        assert "query" in sample, f"Missing 'query' field. Keys: {sample.keys()}"
        assert "label" in sample, f"Missing 'label' field. Keys: {sample.keys()}"

    def test_image_field_is_pil_image(self, chartqa_dataset):
        """Test that the image field contains a PIL Image."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        sample = chartqa_dataset[0]
        image = sample["image"]
        assert isinstance(image, Image.Image), f"Image is {type(image)}, expected PIL Image"

    def test_query_field_is_string(self, chartqa_dataset):
        """Test that query field is a string."""
        sample = chartqa_dataset[0]
        assert isinstance(sample["query"], str)
        assert len(sample["query"]) > 0

    def test_label_field_is_list(self, chartqa_dataset):
        """Test that label field is a list of answers."""
        sample = chartqa_dataset[0]
        label = sample["label"]
        assert isinstance(label, list), f"Label is {type(label)}, expected list"
        assert len(label) > 0, "Label list should not be empty"


class TestChartQATaskConfig:
    """Tests for ChartQA task configuration loading."""

    @pytest.fixture
    def chartqa_config_path(self):
        """Get path to ChartQA config."""
        return Path(__file__).parent.parent / "lm_eval" / "tasks" / "chartqa" / "chartqa.yaml"

    def test_config_file_exists(self, chartqa_config_path):
        """Test that ChartQA config file exists."""
        assert chartqa_config_path.exists(), f"Config not found: {chartqa_config_path}"

    def test_config_loads_successfully(self, chartqa_config_path):
        """Test that TaskConfig loads from YAML."""
        from lm_eval_mini import TaskConfig

        config = TaskConfig.from_yaml(chartqa_config_path)

        assert config.task == "chartqa"
        assert config.dataset_path == "HuggingFaceM4/ChartQA"
        assert config.test_split == "test"

    def test_config_has_multimodal_settings(self, chartqa_config_path):
        """Test that config properly specifies multimodal settings."""
        from lm_eval_mini import TaskConfig

        config = TaskConfig.from_yaml(chartqa_config_path)

        assert config.is_multimodal is True
        assert config.doc_to_image is not None
        assert "image" in config.doc_to_image

    def test_config_has_doc_to_text_template(self, chartqa_config_path):
        """Test that config has proper doc_to_text template."""
        from lm_eval_mini import TaskConfig

        config = TaskConfig.from_yaml(chartqa_config_path)

        assert config.doc_to_text is not None
        assert "<image>" in config.doc_to_text
        assert "{{query}}" in config.doc_to_text
        assert "Final Answer" in config.doc_to_text

    def test_config_has_metrics(self, chartqa_config_path):
        """Test that config has metric definitions."""
        from lm_eval_mini import TaskConfig

        config = TaskConfig.from_yaml(chartqa_config_path)

        assert config.metric_list is not None
        assert len(config.metric_list) > 0


class TestChartQAInstanceBuilding:
    """Tests for building evaluation instances from ChartQA data."""

    @pytest.fixture(scope="class")
    def chartqa_samples(self):
        """Load ChartQA samples."""
        import datasets

        dataset = datasets.load_dataset(
            "HuggingFaceM4/ChartQA",
            split="test",
            streaming=True,
        )
        return list(dataset.take(3))

    @pytest.fixture
    def chartqa_config(self):
        """Load ChartQA config."""
        from lm_eval_mini import TaskConfig

        config_path = Path(__file__).parent.parent / "lm_eval" / "tasks" / "chartqa" / "chartqa.yaml"
        return TaskConfig.from_yaml(config_path)

    def test_build_instances_creates_correct_count(self, chartqa_samples, chartqa_config):
        """Test that build_instances creates correct number of instances."""
        from lm_eval_mini import build_instances

        instances = build_instances(chartqa_config, chartqa_samples)

        assert len(instances) == len(chartqa_samples)

    def test_instances_have_images(self, chartqa_samples, chartqa_config):
        """Test that instances have image data extracted."""
        from lm_eval_mini import build_instances

        instances = build_instances(chartqa_config, chartqa_samples)

        for inst in instances:
            assert len(inst.images) > 0, "Instance should have at least one image"

    def test_instances_have_prompts(self, chartqa_samples, chartqa_config):
        """Test that instances have rendered prompts."""
        from lm_eval_mini import build_instances

        instances = build_instances(chartqa_config, chartqa_samples)

        for i, inst in enumerate(instances):
            assert len(inst.prompt) > 0, f"Instance {i} has empty prompt"
            # Prompt should contain the query from the sample
            assert chartqa_samples[i]["query"] in inst.prompt

    def test_instances_have_targets(self, chartqa_samples, chartqa_config):
        """Test that instances have target values."""
        from lm_eval_mini import build_instances

        instances = build_instances(chartqa_config, chartqa_samples)

        for i, inst in enumerate(instances):
            assert inst.target is not None
            # Target should match first label
            expected = chartqa_samples[i]["label"][0]
            assert inst.target == expected


class TestChartQAMultimodalMessages:
    """Tests for multimodal message building with ChartQA images."""

    @pytest.fixture(scope="class")
    def chartqa_sample(self):
        """Load a single ChartQA sample."""
        import datasets

        dataset = datasets.load_dataset(
            "HuggingFaceM4/ChartQA",
            split="test",
            streaming=True,
        )
        return next(iter(dataset))

    def test_image_encodes_to_base64(self, chartqa_sample):
        """Test that ChartQA images can be encoded to base64."""
        from lm_eval_mini import encode_image_to_base64

        image = chartqa_sample["image"]
        b64 = encode_image_to_base64(image)

        assert len(b64) > 0
        assert isinstance(b64, str)

    def test_multimodal_message_built_correctly(self, chartqa_sample):
        """Test multimodal message structure with real ChartQA image."""
        from lm_eval_mini import build_multimodal_message, encode_image_to_base64

        image = chartqa_sample["image"]
        text = f"<image>{chartqa_sample['query']}\nFinal Answer:"

        messages = build_multimodal_message(text, [image])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content = messages[0]["content"]
        assert len(content) == 2  # image + text

        # First should be image
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64," in content[0]["image_url"]["url"]

        # Second should be text with <image> removed
        assert content[1]["type"] == "text"
        assert "<image>" not in content[1]["text"]
        assert chartqa_sample["query"] in content[1]["text"]


class TestChartQAEndToEnd:
    """End-to-end test with mocked API."""

    @pytest.fixture(scope="class")
    def chartqa_samples(self):
        """Load ChartQA samples."""
        import datasets

        dataset = datasets.load_dataset(
            "HuggingFaceM4/ChartQA",
            split="test",
            streaming=True,
        )
        return list(dataset.take(2))

    def test_full_evaluation_pipeline_with_mock_api(self, chartqa_samples):
        """Test the full evaluation pipeline with mocked API responses."""
        from lm_eval_mini import (
            APIConfig,
            TaskConfig,
            build_instances,
            compute_metrics,
            run_generation,
        )

        # Load config
        config_path = Path(__file__).parent.parent / "lm_eval" / "tasks" / "chartqa" / "chartqa.yaml"
        config = TaskConfig.from_yaml(config_path)

        # Build instances
        instances = build_instances(config, chartqa_samples, limit=2)
        assert len(instances) == 2

        # Mock the API config
        api_config = APIConfig(
            base_url="http://mock-api.test/v1/chat/completions",
            model="mock-model",
            api_key="test-key",
        )

        # Create mock response that matches the expected format
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": f"Final Answer: {instances[0].target}"
                    }
                }
            ]
        }

        async def mock_post(*args, **kwargs):
            class MockResponse:
                ok = True

                async def json(self):
                    return mock_response

            return MockResponse()

        # Mock the aiohttp session
        with patch("lm_eval_mini.aiohttp.ClientSession") as mock_session:
            mock_session_inst = AsyncMock()
            mock_session_inst.post = mock_post
            mock_session.return_value.__aenter__.return_value = mock_session_inst

            # Run generation (async)
            instances = asyncio.run(
                run_generation(instances, api_config, is_multimodal=True)
            )

        # Verify responses were set
        for inst in instances:
            assert inst.response is not None

        # Compute metrics
        metrics = compute_metrics(instances, config)

        assert "relaxed_accuracy" in metrics or "exact_match" in metrics


class TestChartQADatasetFullLoad:
    """Test loading the full ChartQA dataset (non-streaming)."""

    def test_full_dataset_load(self):
        """Test that the full ChartQA dataset can be loaded."""
        import datasets

        # Load full dataset (not streaming)
        dataset = datasets.load_dataset(
            "HuggingFaceM4/ChartQA",
        )

        # Verify splits exist
        assert "test" in dataset, f"No 'test' split. Available: {list(dataset.keys())}"

        test_split = dataset["test"]
        assert len(test_split) > 0

        # Check first sample
        sample = test_split[0]
        assert "image" in sample
        assert "query" in sample
        assert "label" in sample

        print(f"\nChartQA dataset loaded successfully!")
        print(f"  Test split size: {len(test_split)}")
        print(f"  Sample query: {sample['query'][:80]}...")
        print(f"  Sample label: {sample['label']}")
