"""
End-to-end tests for tinyeval CLI.

Tests the full workflow: CLI args → API call → JSON output.
"""

import asyncio
import sys
from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from core import APIConfig, Sample
from tasks import TASKS
from tinyeval import evaluate, main


# Reusable mock sample loaders
def _single_sample(
    max_samples: int | None = None, seed: int | None = None
) -> list[Sample]:
    """Single sample for basic tests."""
    return [Sample(prompt="What is 2+2?", target="4")]


def _multi_sample(
    max_samples: int | None = None, seed: int | None = None
) -> list[Sample]:
    """Multiple samples for concurrency tests."""
    samples = [Sample(prompt=f"Question {i}?", target="42") for i in range(5)]
    if max_samples is not None:
        return samples[:max_samples]
    return samples


class MockResp:
    """Mock aiohttp response."""

    ok = True

    def __init__(self, content: str = "42"):
        self._content = content

    async def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


SamplesFn = Callable[[int | None, int | None], list[Sample]]


@contextmanager
def cli_context(args: list[str], mock_samples: dict[str, SamplesFn]):
    """Context manager for CLI tests: manages sys.argv and TASKS state."""
    original_argv = sys.argv
    original_samples = {name: TASKS[name].samples for name in mock_samples}

    try:
        sys.argv = ["tinyeval"] + args
        for name, samples_fn in mock_samples.items():
            TASKS[name].samples = samples_fn
        yield
    finally:
        sys.argv = original_argv
        for name, samples_fn in original_samples.items():
            TASKS[name].samples = samples_fn


def run_cli_with_mock(args: list[str], mock_samples: dict[str, SamplesFn], post_fn):
    """Run CLI with mocked API post function."""
    with cli_context(args, mock_samples):
        with patch("core.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.post = post_fn
            main()


class TestE2E:
    """End-to-end CLI tests."""

    def test_gsm8k_evaluation(self, capsys):
        """GSM8K task produces correct JSON output."""
        run_cli_with_mock(
            [
                "--tasks",
                "gsm8k_llama",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1"',
                "--max_samples",
                "1",
            ],
            {"gsm8k_llama": _single_sample},
            lambda url, **kwargs: MockResp("The final answer is 4"),
        )

        output = capsys.readouterr().out
        assert '"gsm8k_llama"' in output
        assert '"exact_match": 1.0' in output

    def test_gen_kwargs_passed_to_api(self):
        """gen_kwargs CLI arg flows through to API request payload."""
        captured_payload = None

        def mock_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return MockResp()

        run_cli_with_mock(
            [
                "--tasks",
                "gsm8k_llama",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1"',
                "--max_samples",
                "1",
                "--gen_kwargs",
                'temperature=0.7,max_tokens=100,reasoning_effort="medium"',
            ],
            {"gsm8k_llama": _single_sample},
            mock_post,
        )

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

        class TrackingMockResp(MockResp):
            async def __aenter__(self):
                nonlocal active_requests, max_concurrent_seen
                active_requests += 1
                max_concurrent_seen = max(max_concurrent_seen, active_requests)
                await asyncio.sleep(0.01)
                return await super().__aenter__()

            async def __aexit__(self, *args):
                nonlocal active_requests
                active_requests -= 1

        run_cli_with_mock(
            [
                "--tasks",
                "gsm8k_llama",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1",num_concurrent=2',
            ],
            {"gsm8k_llama": _multi_sample},
            lambda url, **kwargs: TrackingMockResp(),
        )

        assert (
            max_concurrent_seen <= 2
        ), f"Expected max 2 concurrent, saw {max_concurrent_seen}"

    def test_model_args_passed_to_config(self):
        """model_args CLI arg flows through to APIConfig and API request."""
        captured_payload = None

        def mock_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return MockResp()

        run_cli_with_mock(
            [
                "--tasks",
                "gsm8k_llama",
                "--max_samples",
                "1",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1",num_concurrent=4,max_retries=5',
            ],
            {"gsm8k_llama": _single_sample},
            mock_post,
        )

        assert captured_payload["model"] == "test-model"
