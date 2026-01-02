"""
End-to-end tests for tinyeval CLI.

Tests the full workflow: CLI args → API call → JSON output.
"""

import asyncio
import sys
from unittest.mock import patch

import pytest

from core import APIConfig, Sample
from tasks import TASKS
from tinyeval import evaluate, main


def _mock_api_response(content: str):
    """Create mock aiohttp session that returns the given content."""

    class MockResp:
        ok = True

        async def json(self):
            return {"choices": [{"message": {"content": content}}]}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockSession:
        def post(self, url, **kwargs):
            return MockResp()

    return MockSession()


def _run_cli(args: list[str], mock_samples: dict[str, callable], api_response: str):
    """Run tinyeval CLI with mocked API and samples."""
    original_argv = sys.argv
    original_samples = {name: TASKS[name].samples for name in mock_samples}

    try:
        sys.argv = ["tinyeval"] + args
        for name, samples_fn in mock_samples.items():
            TASKS[name].samples = samples_fn

        with patch("core.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = _mock_api_response(
                api_response
            )
            main()
    finally:
        sys.argv = original_argv
        for name, samples_fn in original_samples.items():
            TASKS[name].samples = samples_fn


class TestE2E:
    """End-to-end CLI tests."""

    def test_gsm8k_evaluation(self, capsys):
        """GSM8K task produces correct JSON output."""

        def mock_samples(n):
            yield Sample(prompt="What is 2+2?", target="4")

        _run_cli(
            [
                "--tasks",
                "gsm8k_llama",
                "--model",
                "test-model",
                "--base_url",
                "http://test.com/v1",
                "--max_samples",
                "1",
            ],
            {"gsm8k_llama": mock_samples},
            "The final answer is 4",
        )

        output = capsys.readouterr().out
        assert '"gsm8k_llama"' in output
        assert '"exact_match": 1.0' in output

    def test_gen_kwargs_passed_to_api(self):
        """gen_kwargs CLI arg flows through to API request payload."""
        captured_payload = None

        class MockResp:
            ok = True

            async def json(self):
                return {"choices": [{"message": {"content": "42"}}]}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        def mock_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return MockResp()

        def mock_samples(n):
            yield Sample(prompt="What is 2+2?", target="4")

        original_argv = sys.argv
        original_samples = TASKS["gsm8k_llama"].samples

        try:
            sys.argv = [
                "tinyeval",
                "--tasks",
                "gsm8k_llama",
                "--model",
                "test-model",
                "--base_url",
                "http://test.com/v1",
                "--max_samples",
                "1",
                "--gen_kwargs",
                'temperature=0.7,max_tokens=100,reasoning_effort="medium"',
            ]
            TASKS["gsm8k_llama"].samples = mock_samples

            with patch("core.aiohttp.ClientSession") as mock_session:
                mock_session.return_value.__aenter__.return_value.post = mock_post
                main()
        finally:
            sys.argv = original_argv
            TASKS["gsm8k_llama"].samples = original_samples

        assert captured_payload["temperature"] == 0.7
        assert captured_payload["max_tokens"] == 100
        assert captured_payload["reasoning_effort"] == "medium"

    def test_invalid_task_raises_error(self):
        """Unknown task name raises ValueError."""
        config = APIConfig(url="http://test.com", model="test")
        with pytest.raises(ValueError, match="Unknown task"):
            asyncio.run(evaluate(["nonexistent_task"], config))

    def test_num_concurrent_limits_parallel_requests(self):
        """num_concurrent CLI arg limits the number of parallel API requests."""
        active_requests = 0
        max_concurrent_seen = 0

        class MockResp:
            ok = True

            async def json(self):
                return {"choices": [{"message": {"content": "42"}}]}

            async def __aenter__(self):
                nonlocal active_requests, max_concurrent_seen
                active_requests += 1
                max_concurrent_seen = max(max_concurrent_seen, active_requests)
                await asyncio.sleep(0.01)  # Simulate network delay
                return self

            async def __aexit__(self, *args):
                nonlocal active_requests
                active_requests -= 1

        def mock_post(url, **kwargs):
            return MockResp()

        def mock_samples(n):
            for i in range(5):
                yield Sample(prompt=f"Question {i}?", target="42")

        original_argv = sys.argv
        original_samples = TASKS["gsm8k_llama"].samples

        try:
            sys.argv = [
                "tinyeval",
                "--tasks",
                "gsm8k_llama",
                "--model",
                "test-model",
                "--base_url",
                "http://test.com/v1",
                "--num_concurrent",
                "2",
            ]
            TASKS["gsm8k_llama"].samples = mock_samples

            with patch("core.aiohttp.ClientSession") as mock_session:
                mock_session.return_value.__aenter__.return_value.post = mock_post
                main()
        finally:
            sys.argv = original_argv
            TASKS["gsm8k_llama"].samples = original_samples

        assert (
            max_concurrent_seen <= 2
        ), f"Expected max 2 concurrent, saw {max_concurrent_seen}"

    def test_model_args_passed_to_config(self):
        """model_args CLI arg flows through to APIConfig and API request (lm_eval compat)."""
        captured_payload = None

        class MockResp:
            ok = True

            async def json(self):
                return {"choices": [{"message": {"content": "42"}}]}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        def mock_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return MockResp()

        def mock_samples(n):
            yield Sample(prompt="What is 2+2?", target="4")

        original_argv = sys.argv
        original_samples = TASKS["gsm8k_llama"].samples

        try:
            # lm_eval compatible: model and base_url via model_args only
            sys.argv = [
                "tinyeval",
                "--tasks",
                "gsm8k_llama",
                "--max_samples",
                "1",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1",num_concurrent=4,max_retries=5',
            ]
            TASKS["gsm8k_llama"].samples = mock_samples

            with patch("core.aiohttp.ClientSession") as mock_session:
                mock_session.return_value.__aenter__.return_value.post = mock_post
                main()
        finally:
            sys.argv = original_argv
            TASKS["gsm8k_llama"].samples = original_samples

        assert captured_payload["model"] == "test-model"
