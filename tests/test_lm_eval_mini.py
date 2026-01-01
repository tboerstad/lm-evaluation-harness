"""Tests for the minimal LM evaluation harness implementation.

These tests verify that:
1. HTTP requests are correctly formed
2. The LocalCompletionsAPI class works correctly
3. Both sync and async paths work correctly
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
    from lm_eval_mini import LocalCompletionsAPI

    return LocalCompletionsAPI(
        base_url="http://test-api.example.com/v1/completions",
        tokenizer_backend=None,
        model="gpt-3.5-turbo",
    )


@pytest.fixture
def api_concurrent():
    """Create a LocalCompletionsAPI instance with concurrency enabled."""
    from lm_eval_mini import LocalCompletionsAPI

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
                "lm_eval_mini.aiohttp.TCPConnector", autospec=True
            ) as mock_connector,
            patch(
                "lm_eval_mini.aiohttp.ClientSession", autospec=True
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
        mock_response_data = {
            "choices": [{"text": "Answer", "index": 0, "finish_reason": "stop"}]
        }

        with (
            patch(
                "lm_eval_mini.aiohttp.TCPConnector", autospec=True
            ),
            patch(
                "lm_eval_mini.aiohttp.ClientSession", autospec=True
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

            # Each request should result in one HTTP call
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
        from lm_eval_mini import LocalCompletionsAPI

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


class TestConcurrentRequests:
    """Test concurrent request handling."""

    def test_concurrent_requests_semaphore_limit(self):
        """Test that concurrent requests respect the semaphore limit."""
        from lm_eval_mini import LocalCompletionsAPI

        api_concurrent = LocalCompletionsAPI(
            base_url="http://test-api.example.com/v1/completions",
            tokenizer_backend=None,
            model="gpt-3.5-turbo",
            num_concurrent=2,
        )

        mock_response_data = {
            "choices": [{"text": "Answer", "index": 0, "finish_reason": "stop"}]
        }

        with (
            patch(
                "lm_eval_mini.aiohttp.TCPConnector", autospec=True
            ) as mock_connector,
            patch(
                "lm_eval_mini.aiohttp.ClientSession", autospec=True
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


class TestTaskConfig:
    """Test task configuration loading."""

    def test_task_config_from_dict(self):
        """Test creating TaskConfig from dictionary."""
        from lm_eval_mini import TaskConfig

        config = TaskConfig(
            task="test_task",
            dataset_path="test/dataset",
            output_type="generate_until",
        )

        assert config.task == "test_task"
        assert config.dataset_path == "test/dataset"
        assert config.output_type == "generate_until"

    def test_template_rendering(self):
        """Test jinja2 template rendering."""
        from lm_eval_mini import render_template

        template = "Question: {{question}}\nAnswer:"
        doc = {"question": "What is 2+2?"}

        result = render_template(template, doc)
        assert result == "Question: What is 2+2?\nAnswer:"


class TestMetrics:
    """Test metric computation."""

    def test_exact_match(self):
        """Test exact match metric."""
        from lm_eval_mini import exact_match

        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "world") == 0.0
        assert exact_match("HELLO", "hello", ignore_case=True) == 1.0
        assert exact_match("hello!", "hello", ignore_punctuation=True) == 1.0

    def test_exact_match_with_list(self):
        """Test exact match with multiple references."""
        from lm_eval_mini import exact_match

        assert exact_match("hello", ["hello", "world"]) == 1.0
        assert exact_match("world", ["hello", "world"]) == 1.0
        assert exact_match("foo", ["hello", "world"]) == 0.0

    def test_normalize_text(self):
        """Test text normalization."""
        from lm_eval_mini import normalize_text

        assert normalize_text("  Hello  ") == "Hello"
        assert normalize_text("HELLO", ignore_case=True) == "hello"
        assert normalize_text("hello!", ignore_punctuation=True) == "hello"
