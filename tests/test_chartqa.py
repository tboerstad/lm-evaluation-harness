"""
Tests for ChartQA task.

Tests scoring logic and E2E evaluation without network calls.
"""

import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from core import Sample
from tasks import TASKS
from tasks.chartqa import _relaxed_match, score
from tests.test_tinyeval import MockResp, cli_context, run_cli_with_mock


class TestRelaxedMatch:
    """Tests for ChartQA relaxed matching scoring."""

    def test_exact_match(self):
        """Exact string match scores 1.0."""
        assert _relaxed_match("yes", "yes") == 1.0
        assert _relaxed_match("No", "no") == 1.0

    def test_exact_match_with_final_answer(self):
        """Extracts answer from FINAL ANSWER: prefix."""
        assert _relaxed_match("FINAL ANSWER: yes", "yes") == 1.0
        assert _relaxed_match("Let me think...\nFINAL ANSWER: 42", "42") == 1.0

    def test_numeric_within_tolerance(self):
        """Numbers within 5% tolerance score 1.0."""
        assert _relaxed_match("100", "100") == 1.0
        assert _relaxed_match("105", "100") == 1.0  # exactly 5%
        assert _relaxed_match("95", "100") == 1.0  # exactly -5%
        assert _relaxed_match("102", "100") == 1.0  # within tolerance

    def test_numeric_outside_tolerance(self):
        """Numbers outside 5% tolerance score 0.0."""
        assert _relaxed_match("106", "100") == 0.0  # > 5%
        assert _relaxed_match("94", "100") == 0.0  # < -5%
        assert _relaxed_match("200", "100") == 0.0

    def test_numeric_with_symbols(self):
        """Strips $, %, and commas from numbers."""
        assert _relaxed_match("$100", "100") == 1.0
        assert _relaxed_match("100%", "100") == 1.0
        assert _relaxed_match("$1,000", "1000") == 1.0
        assert _relaxed_match("FINAL ANSWER: $42.50", "42.50") == 1.0

    def test_zero_target(self):
        """Zero target: only exact zero match scores 1.0."""
        assert _relaxed_match("0", "0") == 1.0
        assert _relaxed_match("0.0", "0") == 1.0
        assert _relaxed_match("1", "0") == 0.0

    def test_non_numeric_mismatch(self):
        """Non-matching non-numeric strings score 0.0."""
        assert _relaxed_match("yes", "no") == 0.0
        assert _relaxed_match("apple", "orange") == 0.0

    def test_score_function_delegates(self):
        """score() function delegates to _relaxed_match."""
        assert score("FINAL ANSWER: 100", "100") == 1.0
        assert score("wrong", "right") == 0.0


def _chartqa_sample(_n: int | None) -> Iterator[Sample]:
    """Mock ChartQA sample with multimodal input."""
    mock_image = MagicMock()
    yield Sample(
        prompt=("<image>Question about chart\nFINAL ANSWER:", [mock_image]),
        target="42",
    )


class TestChartQAE2E:
    """End-to-end tests for ChartQA task."""

    def test_chartqa_evaluation(self, capsys):
        """ChartQA task produces correct JSON output with mocked samples."""
        run_cli_with_mock(
            [
                "--tasks",
                "chartqa",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1"',
                "--max_samples",
                "1",
            ],
            {"chartqa": _chartqa_sample},
            lambda url, **kwargs: MockResp("Let me analyze... FINAL ANSWER: 42"),
        )

        output = capsys.readouterr().out
        assert '"chartqa"' in output
        assert '"exact_match": 1.0' in output

    def test_chartqa_numeric_tolerance(self, capsys):
        """ChartQA scoring uses 5% numeric tolerance."""
        run_cli_with_mock(
            [
                "--tasks",
                "chartqa",
                "--model_args",
                'model="test-model",base_url="http://test.com/v1"',
                "--max_samples",
                "1",
            ],
            {"chartqa": _chartqa_sample},
            lambda url, **kwargs: MockResp("FINAL ANSWER: 43"),  # ~2.4% off from 42
        )

        output = capsys.readouterr().out
        assert '"exact_match": 1.0' in output  # within 5% tolerance
