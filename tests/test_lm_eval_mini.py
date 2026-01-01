"""Tests for lm_eval_mini: HTTP requests, image handling, metrics, task config."""

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


@pytest.fixture
def api():
    """LocalCompletionsAPI instance for testing."""
    from lm_eval_mini import LocalCompletionsAPI
    return LocalCompletionsAPI(base_url="http://test-api.example.com/v1/completions", model="gpt-3.5-turbo")


# ============================================================================
# HTTP Request Tests
# ============================================================================

class TestHTTPRequests:
    """Sync and async HTTP request formation."""

    def test_payload_structure_and_stop_sequences(self, api):
        """Verify payload contains model, max_tokens, seed, and stop sequences."""
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(json=lambda: {"choices": [{"text": "42"}]})
            api.model_call(["What is 2+2?"], generate=True, gen_kwargs={"max_tokens": 100, "until": ["Q:", "</s>"]})

            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "gpt-3.5-turbo"
            assert payload["max_tokens"] == 100
            assert payload["seed"] == 1234
            assert payload["stop"] == ["Q:", "</s>"]

    def test_async_concurrency_limit(self):
        """Async requests respect num_concurrent via TCPConnector limit."""
        from lm_eval_mini import LocalCompletionsAPI
        api = LocalCompletionsAPI(base_url="http://test.com/v1/completions", model="gpt-3.5-turbo", num_concurrent=4)

        with patch("lm_eval_mini.aiohttp.TCPConnector") as mock_connector, \
             patch("lm_eval_mini.aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock(ok=True, json=AsyncMock(return_value={"choices": [{"text": "Answer"}]}))
            mock_resp.raise_for_status = lambda: None
            mock_session.return_value.__aenter__.return_value = MagicMock(
                post=lambda *a, **k: DummyAsyncContextManager(mock_resp)
            )
            asyncio.run(api.get_batched_requests(["Q1", "Q2", "Q3"], [("k1",), ("k2",), ("k3",)], generate=True))
            assert mock_connector.call_args[1]["limit"] == 4


# ============================================================================
# Image/Multimodal Tests
# ============================================================================

class TestImageHandling:
    """Image encoding and multimodal message building."""

    def test_encode_pil_image_and_string_passthrough(self):
        """PIL images encode to base64; strings pass through unchanged."""
        from lm_eval_mini import encode_image_to_base64
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
        from lm_eval_mini import build_multimodal_message
        messages = build_multimodal_message("What is in <image>this chart?", ["abc123"])

        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in content[0]["image_url"]["url"]
        assert content[1]["type"] == "text"
        assert "<image>" not in content[1]["text"]

    def test_get_images_from_doc(self):
        """Extract images from doc fields, handling single and multiple fields."""
        from lm_eval_mini import get_images_from_doc
        assert get_images_from_doc({"image": "img1"}, ["image"]) == ["img1"]
        assert get_images_from_doc({"img1": "a", "img2": "b"}, ["img1", "img2"]) == ["a", "b"]

    def test_is_multimodal_detection(self):
        """TaskConfig.is_multimodal based on doc_to_image field."""
        from lm_eval_mini import TaskConfig
        assert TaskConfig(task="gsm8k", dataset_path="gsm8k").is_multimodal is False
        assert TaskConfig(task="chartqa", dataset_path="chartqa", doc_to_image="image").is_multimodal is True


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """exact_match, relaxed_accuracy, anywhere_accuracy."""

    def test_exact_match(self):
        """Basic, case-insensitive, and list reference matching."""
        from lm_eval_mini import exact_match
        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "world") == 0.0
        assert exact_match("HELLO", "hello", ignore_case=True) == 1.0
        assert exact_match("yes", ["yes", "no"]) == 1.0
        assert exact_match("maybe", ["yes", "no"]) == 0.0

    def test_relaxed_accuracy(self):
        """Exact match, 5% numeric tolerance, and Final Answer extraction."""
        from lm_eval_mini import relaxed_accuracy
        assert relaxed_accuracy("42", "42") == 1.0
        assert relaxed_accuracy("Yes", "yes") == 1.0
        assert relaxed_accuracy("42", "40") == 1.0  # 5% tolerance
        assert relaxed_accuracy("50", "40") == 0.0  # >5%
        assert relaxed_accuracy("Final Answer: 42", "42") == 1.0

    def test_anywhere_accuracy(self):
        """Substring match anywhere in prediction."""
        from lm_eval_mini import anywhere_accuracy
        assert anywhere_accuracy("The answer is 42 because...", "42") == 1.0
        assert anywhere_accuracy("I think it's about 50", "42") == 0.0


# ============================================================================
# Chat Payload Tests
# ============================================================================

class TestChatPayload:
    """Chat completions payload construction."""

    def test_payload_structure_and_stop_limit(self):
        """Payload has model, messages, defaults; stop limited to 4."""
        from lm_eval_mini import APIConfig, create_chat_payload
        config = APIConfig(base_url="http://test.com", model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        payload = create_chat_payload(messages, config)
        assert payload["model"] == "gpt-4"
        assert payload["max_tokens"] == 512
        assert payload["temperature"] == 0.0

        payload_with_stop = create_chat_payload(messages, config, gen_kwargs={"until": ["a", "b", "c", "d", "e"]})
        assert len(payload_with_stop["stop"]) == 4  # OpenAI limit


# ============================================================================
# Seed/Reproducibility Tests
# ============================================================================

class TestSeedReproducibility:
    """Seed ensures deterministic few-shot selection."""

    def test_fewshot_same_seed_same_examples(self):
        """Same seed produces identical few-shot examples."""
        import random
        from lm_eval_mini import TaskConfig, get_fewshot_examples

        class MockDataset(dict):
            pass
        dataset = MockDataset()
        dataset["train"] = [{"q": f"Q{i}", "a": f"A{i}"} for i in range(100)]

        config = TaskConfig(task="test", dataset_path="test", training_split="train")

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        examples1 = get_fewshot_examples(config, dataset, 5, rng1)
        examples2 = get_fewshot_examples(config, dataset, 5, rng2)

        assert examples1 == examples2, "Same seed should produce identical examples"

    def test_fewshot_different_seed_different_examples(self):
        """Different seeds produce different few-shot examples."""
        import random
        from lm_eval_mini import TaskConfig, get_fewshot_examples

        class MockDataset(dict):
            pass
        dataset = MockDataset()
        dataset["train"] = [{"q": f"Q{i}", "a": f"A{i}"} for i in range(100)]

        config = TaskConfig(task="test", dataset_path="test", training_split="train")

        rng1 = random.Random(42)
        rng2 = random.Random(99)
        examples1 = get_fewshot_examples(config, dataset, 5, rng1)
        examples2 = get_fewshot_examples(config, dataset, 5, rng2)

        assert examples1 != examples2, "Different seeds should produce different examples"
