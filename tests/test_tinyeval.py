"""End-to-end tests for tinyeval CLI."""

import asyncio
import sys
from unittest.mock import patch

import pytest

from core import APIConfig, Sample, Task
from tasks.gsm8k import _score as gsm8k_score
from tinyeval import evaluate, main


def _single_sample(max_samples: int | None = None) -> list[Sample]:
    """Single sample for basic tests."""
    return [Sample(prompt="What is 2+2?", target="4")]


class MockResp:
    """Mock httpx response."""

    is_success = True

    def __init__(self, content: str = "42"):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def run_cli_with_mock(args: list[str], mock_tasks: dict[str, Task], post_fn):
    """Run CLI with mocked API and tasks."""
    with (
        patch.object(sys, "argv", ["tinyeval"] + args),
        patch.dict("tasks.TASKS", mock_tasks, clear=True),
        patch("core.httpx.AsyncClient") as mock_client,
    ):
        mock_client.return_value.__aenter__.return_value.post = post_fn
        main()


class TestE2E:
    """End-to-end CLI tests."""

    def test_gsm8k_evaluation(self, capsys):
        """GSM8K task produces correct JSON output."""

        async def mock_post(url, **kwargs):
            return MockResp("The final answer is 4")

        mock_tasks = {
            "gsm8k_llama": Task(
                name="gsm8k_llama", samples=_single_sample, score=gsm8k_score
            )
        }

        run_cli_with_mock(
            [
                "--tasks",
                "gsm8k_llama",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1"',
                "--max_samples",
                "1",
            ],
            mock_tasks,
            mock_post,
        )

        output = capsys.readouterr().out
        assert '"gsm8k_llama"' in output
        assert '"exact_match": 1.0' in output

    def test_gen_kwargs_passed_to_api(self):
        """gen_kwargs CLI arg flows through to API request payload."""
        captured_payload = None

        async def mock_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return MockResp()

        mock_tasks = {
            "gsm8k_llama": Task(
                name="gsm8k_llama", samples=_single_sample, score=gsm8k_score
            )
        }

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
            mock_tasks,
            mock_post,
        )

        assert captured_payload["temperature"] == 0.7
        assert captured_payload["max_tokens"] == 100
        assert captured_payload["reasoning_effort"] == "medium"

    def test_invalid_task_raises_error(self):
        """Unknown task name raises ValueError."""
        config = APIConfig(url="http://test.com", model="test", seed=42)
        with pytest.raises(ValueError, match="Unknown task"):
            asyncio.run(evaluate(["nonexistent_task"], config))

    def test_model_args_passed_to_config(self):
        """model_args CLI arg flows through to APIConfig and API request."""
        captured_payload = None

        async def mock_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json")
            return MockResp()

        mock_tasks = {
            "gsm8k_llama": Task(
                name="gsm8k_llama", samples=_single_sample, score=gsm8k_score
            )
        }

        run_cli_with_mock(
            [
                "--tasks",
                "gsm8k_llama",
                "--max_samples",
                "1",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1",num_concurrent=4,max_retries=5',
            ],
            mock_tasks,
            mock_post,
        )

        assert captured_payload["model"] == "test-model"
