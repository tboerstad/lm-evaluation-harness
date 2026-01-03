"""
End-to-end tests for tinyeval CLI.

Tests the full workflow: CLI args → API call → JSON output.
"""

import asyncio
import json
import sys
from unittest.mock import patch

import pytest
from PIL import Image

from core import APIConfig, Sample, Task, _encode_image, compute_task_hash
from tasks.gsm8k import score as gsm8k_score
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
                "model=test-model,base_url=http://test.com/v1",
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
                "model=test-model,base_url=http://test.com/v1",
                "--max_samples",
                "1",
                "--gen_kwargs",
                "temperature=0.7,max_tokens=100,reasoning_effort=medium",
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
                "model=test-model,base_url=http://test.com/v1,num_concurrent=4,max_retries=5",
            ],
            mock_tasks,
            mock_post,
        )

        assert captured_payload["model"] == "test-model"

    def test_output_path_writes_results_json(self, tmp_path):
        """--output_path writes results.json file."""

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
                "model=test-model,base_url=http://test.com/v1",
                "--max_samples",
                "1",
                "--output_path",
                str(tmp_path),
            ],
            mock_tasks,
            mock_post,
        )

        results_file = tmp_path / "results.json"
        assert results_file.exists()

        with open(results_file) as f:
            results = json.load(f)

        assert "gsm8k_llama" in results["results"]
        assert results["results"]["gsm8k_llama"]["metrics"]["exact_match"] == 1.0

    def test_log_samples_writes_jsonl(self, tmp_path):
        """--log_samples writes per-sample JSONL files."""

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
                "model=test-model,base_url=http://test.com/v1",
                "--max_samples",
                "1",
                "--output_path",
                str(tmp_path),
                "--log_samples",
            ],
            mock_tasks,
            mock_post,
        )

        jsonl_file = tmp_path / "samples_gsm8k_llama.jsonl"
        assert jsonl_file.exists()

        with open(jsonl_file) as f:
            sample = json.loads(f.readline())

        assert sample["doc_id"] == 0
        assert sample["target"] == "4"
        assert "prompt" in sample
        assert sample["response"] == "The final answer is 4"
        assert sample["exact_match"] == 1.0


class TestEncodeImage:
    """Tests for _encode_image function."""

    def test_encode_valid_image(self):
        """Valid PIL image encodes to base64."""
        img = Image.new("RGB", (10, 10), color="red")
        result = _encode_image(img)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_passthrough_string(self):
        """Base64 string passes through unchanged."""
        b64_string = "SGVsbG8gV29ybGQ="
        assert _encode_image(b64_string) == b64_string

    def test_encode_rejects_unsupported_type(self):
        """Unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            _encode_image(12345)

    def test_encode_raises_on_corrupt_image(self):
        """Corrupt image raises ValueError."""
        img = Image.new("RGB", (10, 10))
        # Corrupt the image by clearing its internal data
        img.im = None
        with pytest.raises(ValueError, match="Failed to encode image"):
            _encode_image(img)


class TestHashing:
    """Tests for task and dataset hashing."""

    def test_task_hash_deterministic(self):
        """Same samples produce the same hash."""
        samples = [Sample(prompt="What is 2+2?", target="4")]
        hash1 = compute_task_hash(samples)
        hash2 = compute_task_hash(samples)
        assert hash1 == hash2

    def test_task_hash_differs_for_different_samples(self):
        """Different samples produce different hashes."""
        samples1 = [Sample(prompt="What is 2+2?", target="4")]
        samples2 = [Sample(prompt="What is 3+3?", target="6")]
        assert compute_task_hash(samples1) != compute_task_hash(samples2)

    def test_dataset_hash_in_results(self, tmp_path):
        """Evaluation results include dataset_hash."""

        async def mock_post(url, **kwargs):
            return MockResp("4")

        mock_tasks = {
            "test_task": Task(
                name="test_task", samples=_single_sample, score=gsm8k_score
            )
        }

        run_cli_with_mock(
            [
                "--tasks",
                "test_task",
                "--model_args",
                "model=test,base_url=http://test.com/v1",
                "--output_path",
                str(tmp_path),
            ],
            mock_tasks,
            mock_post,
        )

        with open(tmp_path / "results.json") as f:
            results = json.load(f)

        assert "dataset_hash" in results
        assert len(results["dataset_hash"]) == 64  # SHA256 hex length
        assert "task_hash" in results["results"]["test_task"]
