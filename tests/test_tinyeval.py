"""Tests for tinyeval: HTTP requests, image handling, metrics, task configs."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class DummyAsyncContextManager:
    """Mock async context manager for aiohttp responses."""
    def __init__(self, result):
        self.result = result
    async def __aenter__(self):
        return self.result
    async def __aexit__(self, *args):
        pass


class TestImageHandling:
    """Image encoding and multimodal message building."""

    def test_encode_pil_image_and_string_passthrough(self):
        """PIL images encode to base64; strings pass through unchanged."""
        from tinyeval import encode_image_to_base64
        try:
            from PIL import Image
            img = Image.new("RGB", (10, 10), color="red")
            b64 = encode_image_to_base64(img)
            assert base64.b64decode(b64)  # Valid base64
        except ImportError:
            pytest.skip("PIL not available")

        # Strings pass through
        url = "https://example.com/image.png"
        assert encode_image_to_base64(url) == url

    def test_multimodal_message_structure(self):
        """Vision API message: images first, text second, <image> placeholder removed."""
        from tinyeval import build_multimodal_message
        messages = build_multimodal_message("What is in <image>this chart?", ["abc123"])

        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in content[0]["image_url"]["url"]
        assert content[1]["type"] == "text"
        assert "<image>" not in content[1]["text"]

    def test_get_images_from_doc(self):
        """Extract images from doc fields, handling single and multiple fields."""
        from tinyeval import get_images_from_doc
        assert get_images_from_doc({"image": "img1"}, ["image"]) == ["img1"]
        assert get_images_from_doc({"img1": "a", "img2": "b"}, ["img1", "img2"]) == ["a", "b"]

    def test_is_multimodal_detection(self):
        """TaskConfig.is_multimodal based on doc_to_image field."""
        from tinyeval import TaskConfig
        assert TaskConfig(task="gsm8k", dataset_path="gsm8k", doc_to_text=lambda x: "", doc_to_target=lambda x: "").is_multimodal is False
        assert TaskConfig(task="chartqa", dataset_path="chartqa", doc_to_text=lambda x: "", doc_to_target=lambda x: "", doc_to_image=["image"]).is_multimodal is True


class TestMetrics:
    """exact_match, relaxed_accuracy."""

    def test_exact_match(self):
        """Basic matching with normalization."""
        from tinyeval import exact_match
        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "world") == 0.0
        assert exact_match("HELLO", "hello") == 1.0  # case insensitive by default
        assert exact_match("42", "42") == 1.0
        assert exact_match("$42", "42") == 1.0  # $ is stripped

    def test_relaxed_accuracy(self):
        """Exact match, 5% numeric tolerance, and FINAL ANSWER extraction."""
        from tinyeval import relaxed_accuracy
        assert relaxed_accuracy("42", "42") == 1.0
        assert relaxed_accuracy("Yes", "yes") == 1.0
        assert relaxed_accuracy("42", "40") == 1.0  # 5% tolerance
        assert relaxed_accuracy("50", "40") == 0.0  # >5%
        assert relaxed_accuracy("FINAL ANSWER: 42", "42") == 1.0


class TestChatPayload:
    """Chat completions payload construction."""

    def test_payload_structure_and_stop_limit(self):
        """Payload has model, messages, defaults; stop limited to 4."""
        from tinyeval import APIConfig, TaskConfig, create_chat_payload
        api_config = APIConfig(base_url="http://test.com", model="gpt-4")
        task_config = TaskConfig(
            task="test",
            dataset_path="test",
            doc_to_text=lambda x: "",
            doc_to_target=lambda x: "",
            stop_sequences=["a", "b", "c", "d", "e"],
        )
        messages = [{"role": "user", "content": "Hello"}]

        payload = create_chat_payload(messages, api_config, task_config)
        assert payload["model"] == "gpt-4"
        assert payload["max_tokens"] == 512
        assert payload["temperature"] == 0.0
        assert len(payload["stop"]) == 4  # OpenAI limit


class TestBuiltInTasks:
    """Built-in task configs (gsm8k_llama, chartqa)."""

    def test_gsm8k_llama_config(self):
        """GSM8K Llama config has correct structure."""
        from tinyeval import TASKS
        config = TASKS["gsm8k_llama"]
        assert config.task == "gsm8k_llama"
        assert config.dataset_path == "gsm8k"
        assert config.fewshot_examples is not None
        assert len(config.fewshot_examples) == 8
        assert not config.is_multimodal

    def test_chartqa_config(self):
        """ChartQA config has correct structure."""
        from tinyeval import TASKS
        config = TASKS["chartqa"]
        assert config.task == "chartqa"
        assert config.dataset_path == "HuggingFaceM4/ChartQA"
        assert config.is_multimodal
        assert config.doc_to_image == ["image"]

    def test_gsm8k_doc_to_text(self):
        """GSM8K doc_to_text generates correct prompt."""
        from tinyeval import gsm8k_doc_to_text
        doc = {"question": "What is 2 + 2?"}
        prompt = gsm8k_doc_to_text(doc)
        assert "What is 2 + 2?" in prompt
        assert "The final answer is" in prompt

    def test_gsm8k_doc_to_target(self):
        """GSM8K doc_to_target extracts answer after ####."""
        from tinyeval import gsm8k_doc_to_target
        assert gsm8k_doc_to_target({"answer": "Some reasoning #### 42"}) == "42"
        assert gsm8k_doc_to_target({"target": "direct answer"}) == "direct answer"

    def test_gsm8k_extract_answer(self):
        """GSM8K answer extraction from model response."""
        from tinyeval import gsm8k_extract_answer
        assert gsm8k_extract_answer("The final answer is 42") == "42"
        assert gsm8k_extract_answer("The final answer is $1,234") == "$1,234"
        assert gsm8k_extract_answer("Some text with 42 in it") == "42"

    def test_chartqa_doc_to_text(self):
        """ChartQA doc_to_text generates correct prompt."""
        from tinyeval import chartqa_doc_to_text
        doc = {"query": "What is the total revenue?"}
        prompt = chartqa_doc_to_text(doc)
        assert "What is the total revenue?" in prompt
        assert "FINAL ANSWER:" in prompt
        assert "<image>" in prompt

    def test_chartqa_doc_to_target(self):
        """ChartQA doc_to_target extracts first label."""
        from tinyeval import chartqa_doc_to_target
        assert chartqa_doc_to_target({"label": ["42", "forty-two"]}) == "42"
        assert chartqa_doc_to_target({"label": "single"}) == "single"


class TestInstanceBuilding:
    """Instance building from documents."""

    def test_build_instances_text_task(self):
        """Build instances for text-only task."""
        from tinyeval import TASKS, build_instances

        config = TASKS["gsm8k_llama"]
        docs = [
            {"question": "What is 1 + 1?", "answer": "1 + 1 = 2 #### 2"},
            {"question": "What is 2 + 2?", "answer": "2 + 2 = 4 #### 4"},
        ]
        instances = build_instances(config, docs)

        assert len(instances) == 2
        assert instances[0].target == "2"
        assert instances[1].target == "4"
        assert "What is 1 + 1?" in instances[0].prompt
        assert instances[0].images == []

    def test_build_fewshot_context(self):
        """Few-shot context is built from examples."""
        from tinyeval import TASKS, build_fewshot_context

        config = TASKS["gsm8k_llama"]
        context = build_fewshot_context(config)

        assert "15 trees" in context  # First example question
        assert "The final answer is 6" in context  # First example answer


class TestGeneration:
    """Generation with mocked API."""

    def test_run_generation_populates_responses(self):
        """run_generation populates instance responses."""
        from tinyeval import APIConfig, Instance, TaskConfig, run_generation

        instances = [
            Instance(doc={}, doc_id=0, prompt="Test prompt", target="42"),
        ]
        api_config = APIConfig(base_url="http://test.com", model="gpt-4")
        task_config = TaskConfig(
            task="test",
            dataset_path="test",
            doc_to_text=lambda x: "",
            doc_to_target=lambda x: "",
        )

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
            instances = asyncio.run(run_generation(instances, api_config, task_config))

        assert instances[0].response == "The answer is 42"


class TestMetricsComputation:
    """Metrics computation over instances."""

    def test_compute_metrics_gsm8k(self):
        """Compute metrics for GSM8K-like instances."""
        from tinyeval import Instance, TASKS, compute_metrics

        config = TASKS["gsm8k_llama"]
        instances = [
            Instance(doc={}, doc_id=0, prompt="", target="42", response="The final answer is 42"),
            Instance(doc={}, doc_id=1, prompt="", target="100", response="The final answer is 100"),
            Instance(doc={}, doc_id=2, prompt="", target="50", response="The final answer is 60"),  # Wrong
        ]

        metrics = compute_metrics(instances, config)
        assert metrics["exact_match"] == pytest.approx(2/3)

    def test_compute_metrics_chartqa(self):
        """Compute metrics for ChartQA-like instances."""
        from tinyeval import Instance, TASKS, compute_metrics

        config = TASKS["chartqa"]
        instances = [
            Instance(doc={}, doc_id=0, prompt="", target="42", response="FINAL ANSWER: 42"),
            Instance(doc={}, doc_id=1, prompt="", target="100", response="FINAL ANSWER: 95"),  # Within 5%
            Instance(doc={}, doc_id=2, prompt="", target="50", response="FINAL ANSWER: 80"),  # Wrong
        ]

        metrics = compute_metrics(instances, config)
        assert metrics["relaxed_accuracy"] == pytest.approx(2/3)
