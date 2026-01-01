"""Tests for CLI HTTP request handling (sync and async).

These tests verify that:
1. HTTP requests are correctly formed
2. The right number of requests are sent
3. Both sync and async paths work correctly
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lm_eval.models.openai_completions import LocalCompletionsAPI


class DummyAsyncContextManager:
    """Helper class for mocking async context managers."""

    def __init__(self, result):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.fixture
def mock_openai_response():
    """Standard OpenAI API response for generation."""
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "text": "The answer is 42.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def api():
    """Create a LocalCompletionsAPI instance for testing."""
    return LocalCompletionsAPI(
        base_url="http://test-api.example.com/v1/completions",
        tokenizer_backend=None,
        model="gpt-3.5-turbo",
    )


@pytest.fixture
def api_concurrent():
    """Create a LocalCompletionsAPI instance with concurrency enabled."""
    return LocalCompletionsAPI(
        base_url="http://test-api.example.com/v1/completions",
        tokenizer_backend=None,
        model="gpt-3.5-turbo",
        num_concurrent=4,
    )


class TestSyncHttpRequests:
    """Test synchronous HTTP request handling."""

    def test_single_generate_request_payload(self, api, mock_openai_response):
        """Test that a single generate request has the correct payload format."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            result = api.model_call(
                messages=["What is 2 + 2?"],
                generate=True,
                gen_kwargs={"max_tokens": 100, "temperature": 0.0},
            )

            # Verify the request was made
            mock_post.assert_called_once()

            # Verify the payload structure
            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            assert "prompt" in payload
            assert "model" in payload
            assert payload["model"] == "gpt-3.5-turbo"
            assert "max_tokens" in payload
            assert payload["max_tokens"] == 100
            assert payload["temperature"] == 0.0

    def test_generate_request_with_stop_sequences(self, api, mock_openai_response):
        """Test that stop sequences are correctly passed in the payload."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            api.model_call(
                messages=["Question: What is 5 + 5?\nAnswer:"],
                generate=True,
                gen_kwargs={
                    "max_tokens": 256,
                    "until": ["Question:", "</s>"],
                },
            )

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            assert "stop" in payload
            assert payload["stop"] == ["Question:", "</s>"]

    def test_multiple_sync_requests_count(self, api, mock_openai_response):
        """Test that multiple synchronous requests are made correctly."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            # Make 5 individual requests
            num_requests = 5
            for i in range(num_requests):
                api.model_call(
                    messages=[f"Question {i}: What is {i} + {i}?\nAnswer:"],
                    generate=True,
                    gen_kwargs={"max_tokens": 100},
                )

            assert mock_post.call_count == num_requests

    def test_request_url_and_headers(self, api, mock_openai_response):
        """Test that URL and headers are correctly set."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            api.model_call(
                messages=["Test prompt"],
                generate=True,
                gen_kwargs={},
            )

            args, kwargs = mock_post.call_args
            assert args[0] == "http://test-api.example.com/v1/completions"
            assert "headers" in kwargs


class TestAsyncHttpRequests:
    """Test asynchronous HTTP request handling."""

    def test_async_batched_requests_structure(
        self, api_concurrent, mock_openai_response
    ):
        """Test that async batched requests are correctly structured."""
        with (
            patch(
                "lm_eval.models.api_models.TCPConnector", autospec=True
            ) as mock_connector,
            patch(
                "lm_eval.models.api_models.ClientSession", autospec=True
            ) as mock_client_session,
        ):
            mock_session_instance = AsyncMock()
            mock_post_response = AsyncMock()
            mock_post_response.status = 200
            mock_post_response.ok = True
            mock_post_response.json = AsyncMock(return_value=mock_openai_response)
            mock_post_response.raise_for_status = lambda: None

            # Track all payloads sent
            captured_payloads = []

            def capture_post(*args, **kwargs):
                if "json" in kwargs:
                    captured_payloads.append(kwargs["json"])
                return DummyAsyncContextManager(mock_post_response)

            mock_session_instance.post = capture_post
            mock_client_session.return_value.__aenter__.return_value = (
                mock_session_instance
            )

            # Test with 3 requests
            requests = [
                "Question 1: What is 2+2?",
                "Question 2: What is 3+3?",
                "Question 3: What is 4+4?",
            ]
            cache_keys = [("key1",), ("key2",), ("key3",)]

            async def run():
                return await api_concurrent.get_batched_requests(
                    requests,
                    cache_keys,
                    generate=True,
                    gen_kwargs={"max_tokens": 100},
                )

            asyncio.run(run())

            # Verify connector was created with correct concurrency limit
            mock_connector.assert_called_once()
            connector_call_kwargs = mock_connector.call_args[1]
            assert connector_call_kwargs["limit"] == 4

            # Verify payloads were captured
            assert len(captured_payloads) == 3
            for payload in captured_payloads:
                assert "model" in payload
                assert "max_tokens" in payload

    def test_async_requests_count_matches_input(self, api_concurrent):
        """Test that the number of async requests matches the input."""
        # Response must have index field for parse_generations
        mock_response_data = {
            "choices": [{"text": "Answer", "index": 0, "finish_reason": "stop"}]
        }

        with (
            patch(
                "lm_eval.models.api_models.TCPConnector", autospec=True
            ),
            patch(
                "lm_eval.models.api_models.ClientSession", autospec=True
            ) as mock_client_session,
        ):
            mock_session_instance = AsyncMock()
            mock_post_response = AsyncMock()
            mock_post_response.status = 200
            mock_post_response.ok = True
            mock_post_response.json = AsyncMock(return_value=mock_response_data)
            mock_post_response.raise_for_status = lambda: None

            request_count = 0

            def capture_post(*args, **kwargs):
                nonlocal request_count
                request_count += 1
                return DummyAsyncContextManager(mock_post_response)

            mock_session_instance.post = capture_post
            mock_client_session.return_value.__aenter__.return_value = (
                mock_session_instance
            )

            # Test with 7 requests (not a round number)
            num_requests = 7
            requests = [f"Question {i}" for i in range(num_requests)]
            cache_keys = [(f"key{i}",) for i in range(num_requests)]

            async def run():
                return await api_concurrent.get_batched_requests(
                    requests,
                    cache_keys,
                    generate=True,
                    gen_kwargs={"max_tokens": 50},
                )

            asyncio.run(run())

            # Each request should result in one HTTP call (batch_size=1 by default)
            assert request_count == num_requests


class TestGsm8kStyleRequests:
    """Test requests similar to gsm8k task format."""

    def test_gsm8k_style_generation_request(self, api, mock_openai_response):
        """Test a request formatted like gsm8k task."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            # gsm8k-style prompt
            prompt = "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:"

            api.model_call(
                messages=[prompt],
                generate=True,
                gen_kwargs={
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "until": ["Question:", "</s>", "<|im_end|>"],
                    "do_sample": False,
                },
            )

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            # Verify gsm8k-specific settings
            assert payload["temperature"] == 0.0
            assert "Question:" in payload["stop"]


class TestChartQAStyleRequests:
    """Test requests similar to chartqa task format (multimodal)."""

    def test_chartqa_style_generation_kwargs(self, api, mock_openai_response):
        """Test generation kwargs similar to chartqa task."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            # chartqa-style prompt (without actual image for this test)
            prompt = "What is the percentage shown in the pie chart?\nAnalyze the image and question carefully."

            api.model_call(
                messages=[prompt],
                generate=True,
                gen_kwargs={
                    "max_gen_toks": 512,
                    "temperature": 0.0,
                    "do_sample": False,
                    "until": [],
                },
            )

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            # Verify chartqa-specific settings
            assert payload["temperature"] == 0.0
            assert payload["max_tokens"] == 512


class TestRequestPayloadValidation:
    """Test that request payloads are correctly validated."""

    def test_seed_is_passed_in_payload(self, api, mock_openai_response):
        """Test that the seed is correctly passed in the payload."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            api.model_call(
                messages=["Test prompt"],
                generate=True,
                gen_kwargs={},
            )

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            # Default seed should be 1234
            assert "seed" in payload
            assert payload["seed"] == 1234

    def test_model_name_in_payload(self, mock_openai_response):
        """Test that model name is correctly set in payload."""
        custom_model = LocalCompletionsAPI(
            base_url="http://test-api.example.com/v1/completions",
            tokenizer_backend=None,
            model="custom-model-name",
        )

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            custom_model.model_call(
                messages=["Test"],
                generate=True,
                gen_kwargs={},
            )

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]

            assert payload["model"] == "custom-model-name"


class TestBatchedRequestsCounting:
    """Test that batched requests are counted correctly."""

    def test_batched_sync_requests_with_batch_size(self, mock_openai_response):
        """Test sync requests with batch_size > 1."""
        api_batched = LocalCompletionsAPI(
            base_url="http://test-api.example.com/v1/completions",
            tokenizer_backend=None,
            model="gpt-3.5-turbo",
            batch_size=2,
        )

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_openai_response
            mock_response.ok = True
            mock_post.return_value = mock_response

            # Make requests - with batch_size=2, prompts should be batched
            prompts = ["Prompt 1", "Prompt 2"]
            api_batched.model_call(
                messages=prompts,
                generate=True,
                gen_kwargs={"max_tokens": 100},
            )

            # Should be one call with batched prompts
            assert mock_post.call_count == 1

            _, kwargs = mock_post.call_args
            payload = kwargs["json"]
            # The prompt should be a list of prompts for batching
            assert "prompt" in payload

    def test_concurrent_requests_semaphore_limit(self):
        """Test that concurrent requests respect the semaphore limit."""
        api_concurrent = LocalCompletionsAPI(
            base_url="http://test-api.example.com/v1/completions",
            tokenizer_backend=None,
            model="gpt-3.5-turbo",
            num_concurrent=2,
        )

        # Response must have index field for parse_generations
        mock_response_data = {
            "choices": [{"text": "Answer", "index": 0, "finish_reason": "stop"}]
        }

        with (
            patch(
                "lm_eval.models.api_models.TCPConnector", autospec=True
            ) as mock_connector,
            patch(
                "lm_eval.models.api_models.ClientSession", autospec=True
            ) as mock_client_session,
        ):
            mock_session_instance = AsyncMock()
            mock_post_response = AsyncMock()
            mock_post_response.status = 200
            mock_post_response.ok = True
            mock_post_response.json = AsyncMock(return_value=mock_response_data)
            mock_post_response.raise_for_status = lambda: None

            mock_session_instance.post = lambda *args, **kwargs: DummyAsyncContextManager(
                mock_post_response
            )
            mock_client_session.return_value.__aenter__.return_value = (
                mock_session_instance
            )

            requests = ["Q1", "Q2", "Q3", "Q4"]
            cache_keys = [("k1",), ("k2",), ("k3",), ("k4",)]

            async def run():
                return await api_concurrent.get_batched_requests(
                    requests,
                    cache_keys,
                    generate=True,
                    gen_kwargs={},
                )

            asyncio.run(run())

            # Verify TCPConnector was created with correct limit
            mock_connector.assert_called_once()
            assert mock_connector.call_args[1]["limit"] == 2
