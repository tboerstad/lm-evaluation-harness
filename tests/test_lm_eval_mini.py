"""Tests for the minimal LM evaluation harness implementation.

Covers:
1. HTTP request formation (sync and async)
2. Image/multimodal handling
3. Metrics (exact_match, relaxed_accuracy)
4. Task configuration
"""

import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class DummyAsyncContextManager:
    """Helper for mocking async context managers."""

    def __init__(self, result):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_openai_response():
    """Standard OpenAI chat completion response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4",
        "choices": [{"index": 0, "message": {"content": "42"}, "finish_reason": "stop"}],
    }


@pytest.fixture
def api():
    """Create a LocalCompletionsAPI instance."""
    from lm_eval_mini import LocalCompletionsAPI

    return LocalCompletionsAPI(
        base_url="http://test-api.example.com/v1/completions",
        model="gpt-3.5-turbo",
    )


# ============================================================================
# HTTP Request Tests
# ============================================================================


class TestSyncRequests:
    def test_payload_structure(self, api, mock_openai_response):
        """Test request payload has correct structure."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            api.model_call(
                messages=["What is 2+2?"],
                generate=True,
                gen_kwargs={"max_tokens": 100, "temperature": 0.0},
            )

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            assert payload["model"] == "gpt-3.5-turbo"
            assert payload["max_tokens"] == 100
            assert payload["seed"] == 1234

    def test_stop_sequences(self, api, mock_openai_response):
        """Test stop sequences are passed correctly."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_post.return_value = mock_response

            api.model_call(
                messages=["Q: What is 5+5?\nA:"],
                generate=True,
                gen_kwargs={"until": ["Q:", "</s>"]},
            )

            payload = mock_post.call_args[1]["json"]
            assert payload["stop"] == ["Q:", "</s>"]


class TestAsyncRequests:
    def test_concurrent_limit(self):
        """Test async requests respect concurrency limit."""
        from lm_eval_mini import LocalCompletionsAPI

        api = LocalCompletionsAPI(
            base_url="http://test.com/v1/completions",
            model="gpt-3.5-turbo",
            num_concurrent=4,
        )

        mock_response = {"choices": [{"text": "Answer", "index": 0}]}

        with (
            patch("lm_eval_mini.aiohttp.TCPConnector") as mock_connector,
            patch("lm_eval_mini.aiohttp.ClientSession") as mock_session,
        ):
            mock_session_inst = AsyncMock()
            mock_post_resp = AsyncMock()
            mock_post_resp.ok = True
            mock_post_resp.json = AsyncMock(return_value=mock_response)
            mock_post_resp.raise_for_status = lambda: None
            mock_session_inst.post = lambda *a, **k: DummyAsyncContextManager(mock_post_resp)
            mock_session.return_value.__aenter__.return_value = mock_session_inst

            asyncio.run(
                api.get_batched_requests(
                    ["Q1", "Q2", "Q3"],
                    [("k1",), ("k2",), ("k3",)],
                    generate=True,
                )
            )

            assert mock_connector.call_args[1]["limit"] == 4


# ============================================================================
# Image/Multimodal Tests
# ============================================================================


class TestImageHandling:
    def test_encode_pil_image(self):
        """Test PIL image encoding to base64."""
        from lm_eval_mini import encode_image_to_base64

        try:
            from PIL import Image

            # Create a simple 10x10 red image
            img = Image.new("RGB", (10, 10), color="red")
            b64 = encode_image_to_base64(img)

            assert len(b64) > 0
            # Verify it's valid base64
            decoded = base64.b64decode(b64)
            assert len(decoded) > 0
        except ImportError:
            pytest.skip("PIL not available")

    def test_encode_string_passthrough(self):
        """Test that string URLs/base64 pass through unchanged."""
        from lm_eval_mini import encode_image_to_base64

        url = "https://example.com/image.png"
        assert encode_image_to_base64(url) == url

        existing_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        assert encode_image_to_base64(existing_b64) == existing_b64

    def test_multimodal_message_structure(self):
        """Test multimodal message has correct structure for vision API."""
        from lm_eval_mini import build_multimodal_message

        # Test with base64 string (simulating encoded image)
        fake_b64 = "abc123"
        messages = build_multimodal_message("What is in <image>this chart?", [fake_b64])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content = messages[0]["content"]
        assert len(content) == 2  # 1 image + 1 text

        # Image should be first
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in content[0]["image_url"]["url"]

        # Text should be second, with <image> removed
        assert content[1]["type"] == "text"
        assert "<image>" not in content[1]["text"]
        assert "What is in this chart?" in content[1]["text"]

    def test_multimodal_message_no_images(self):
        """Test multimodal message with empty image list."""
        from lm_eval_mini import build_multimodal_message

        messages = build_multimodal_message("Just text", [])
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0]["type"] == "text"

    def test_get_images_from_doc(self):
        """Test extracting images from document."""
        from lm_eval_mini import get_images_from_doc

        doc = {
            "image": "img1",
            "other_field": "not_an_image",
        }
        images = get_images_from_doc(doc, ["image"])
        assert images == ["img1"]

        # Multiple image fields
        doc2 = {"img1": "a", "img2": "b"}
        images2 = get_images_from_doc(doc2, ["img1", "img2"])
        assert images2 == ["a", "b"]


class TestTaskConfigMultimodal:
    def test_is_multimodal_detection(self):
        """Test task correctly detects multimodal configuration."""
        from lm_eval_mini import TaskConfig

        # Text-only task
        text_task = TaskConfig(task="gsm8k", dataset_path="gsm8k")
        assert text_task.is_multimodal is False

        # Multimodal task (single field)
        mm_task = TaskConfig(task="chartqa", dataset_path="chartqa", doc_to_image="image")
        assert mm_task.is_multimodal is True

        # Multimodal task (list of fields)
        mm_task2 = TaskConfig(
            task="multi", dataset_path="multi", doc_to_image=["img1", "img2"]
        )
        assert mm_task2.is_multimodal is True


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    def test_exact_match_basic(self):
        """Test basic exact match."""
        from lm_eval_mini import exact_match

        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "world") == 0.0

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive exact match."""
        from lm_eval_mini import exact_match

        assert exact_match("HELLO", "hello", ignore_case=True) == 1.0
        assert exact_match("Hello World", "hello world", ignore_case=True) == 1.0

    def test_exact_match_with_list(self):
        """Test exact match against multiple references."""
        from lm_eval_mini import exact_match

        assert exact_match("yes", ["yes", "no"]) == 1.0
        assert exact_match("maybe", ["yes", "no"]) == 0.0

    def test_relaxed_accuracy_exact(self):
        """Test relaxed accuracy with exact matches."""
        from lm_eval_mini import relaxed_accuracy

        assert relaxed_accuracy("42", "42") == 1.0
        assert relaxed_accuracy("Yes", "yes") == 1.0

    def test_relaxed_accuracy_numeric_tolerance(self):
        """Test relaxed accuracy with 5% tolerance."""
        from lm_eval_mini import relaxed_accuracy

        # Within 5% tolerance
        assert relaxed_accuracy("42", "40") == 1.0  # 5% of 40 = 2
        assert relaxed_accuracy("105", "100") == 1.0  # 5% of 100 = 5

        # Outside 5% tolerance
        assert relaxed_accuracy("50", "40") == 0.0  # 25% difference
        assert relaxed_accuracy("120", "100") == 0.0  # 20% difference

    def test_relaxed_accuracy_final_answer_format(self):
        """Test relaxed accuracy extracts from 'Final Answer:' format."""
        from lm_eval_mini import relaxed_accuracy

        assert relaxed_accuracy("Final Answer: 42", "42") == 1.0
        assert relaxed_accuracy("Let me think... Final Answer: Yes", "Yes") == 1.0

    def test_anywhere_accuracy(self):
        """Test anywhere accuracy finds substring."""
        from lm_eval_mini import anywhere_accuracy

        assert anywhere_accuracy("The answer is 42 because...", "42") == 1.0
        assert anywhere_accuracy("I think it's about 50", "42") == 0.0
        assert anywhere_accuracy("YES, that is correct", "yes") == 1.0


# ============================================================================
# Chat Payload Tests
# ============================================================================


class TestChatPayload:
    def test_basic_payload(self):
        """Test basic chat payload structure."""
        from lm_eval_mini import APIConfig, create_chat_payload

        config = APIConfig(base_url="http://test.com", model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        payload = create_chat_payload(messages, config)

        assert payload["model"] == "gpt-4"
        assert payload["messages"] == messages
        assert payload["max_tokens"] == 512  # default
        assert payload["temperature"] == 0.0

    def test_payload_with_stop(self):
        """Test payload includes stop sequences."""
        from lm_eval_mini import APIConfig, create_chat_payload

        config = APIConfig(base_url="http://test.com", model="gpt-4")
        payload = create_chat_payload(
            [{"role": "user", "content": "Hi"}],
            config,
            gen_kwargs={"until": ["stop1", "stop2", "stop3", "stop4", "stop5"]},
        )

        # Should limit to 4 stop sequences (OpenAI limit)
        assert len(payload["stop"]) == 4
