"""Test running a single question from multiple tasks.

This test verifies that the evaluation pipeline correctly:
1. Loads task configurations for triviaqa, openbookqa, and piqa
2. Builds instances from a single question per task
3. Runs generation with mocked API responses
4. Computes metrics for each task
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

TASKS_DIR = Path(__file__).parent.parent / "tasks"


class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, content: str):
        self.ok = True
        self._content = content

    async def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class MockContextManager:
    """Mock async context manager for aiohttp post."""

    def __init__(self, response: MockResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class TestSingleQuestionMultiTask:
    """Test running a single question from triviaqa, openbookqa, and piqa."""

    @pytest.fixture
    def triviaqa_config(self):
        """Load triviaqa task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "triviaqa" / "default.yaml")

    @pytest.fixture
    def openbookqa_config(self):
        """Load openbookqa task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "openbookqa" / "openbookqa.yaml")

    @pytest.fixture
    def piqa_config(self):
        """Load piqa task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "piqa" / "piqa.yaml")

    def test_triviaqa_single_question(self, triviaqa_config):
        """Test running a single TriviaQA trivia question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock TriviaQA document
        doc = {
            "question": "What is the capital of France",
            "answer": {
                "aliases": ["Paris", "paris", "PARIS"],
                "normalized_aliases": ["paris"],
                "value": "Paris",
            },
        }

        instances = build_instances(triviaqa_config, [doc])
        assert len(instances) == 1
        assert "capital of France" in instances[0].prompt
        assert instances[0].target is not None

        # Mock API response with correct answer
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("Paris")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        assert "Paris" in instances[0].response

        metrics = compute_metrics(instances, triviaqa_config)
        assert "exact_match" in metrics

    def test_openbookqa_single_question(self, openbookqa_config):
        """Test running a single OpenBookQA science question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock OpenBookQA document
        doc = {
            "question_stem": "Which of the following is a renewable energy source?",
            "choices": {
                "text": ["Coal", "Natural gas", "Solar power", "Nuclear power"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        }

        instances = build_instances(openbookqa_config, [doc])
        assert len(instances) == 1
        assert "renewable" in instances[0].prompt

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("C")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        metrics = compute_metrics(instances, openbookqa_config)
        assert len(metrics) > 0

    def test_piqa_single_question(self, piqa_config):
        """Test running a single PIQA physical intuition question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock PIQA document
        doc = {
            "goal": "How do you make ice cubes?",
            "sol1": "Put water in a freezer",
            "sol2": "Put water in an oven",
            "label": 0,  # First solution is correct
        }

        instances = build_instances(piqa_config, [doc])
        assert len(instances) == 1
        assert "ice cubes" in instances[0].prompt

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("0")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        metrics = compute_metrics(instances, piqa_config)
        assert len(metrics) > 0

    def test_all_three_tasks_together(self, triviaqa_config, openbookqa_config, piqa_config):
        """Test running all three tasks together in a single pipeline."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create mock documents for each task
        triviaqa_doc = {
            "question": "Who wrote Romeo and Juliet",
            "answer": {
                "aliases": ["Shakespeare", "William Shakespeare"],
                "normalized_aliases": ["shakespeare", "william shakespeare"],
                "value": "William Shakespeare",
            },
        }
        openbookqa_doc = {
            "question_stem": "What do plants need to perform photosynthesis?",
            "choices": {
                "text": ["Darkness", "Sunlight", "Cold", "Wind"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        }
        piqa_doc = {
            "goal": "How do you cut paper?",
            "sol1": "Use scissors",
            "sol2": "Use a pillow",
            "label": 0,
        }

        # Build instances for each task
        triviaqa_instances = build_instances(triviaqa_config, [triviaqa_doc])
        openbookqa_instances = build_instances(openbookqa_config, [openbookqa_doc])
        piqa_instances = build_instances(piqa_config, [piqa_doc])

        all_instances = triviaqa_instances + openbookqa_instances + piqa_instances
        assert len(all_instances) == 3

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )

        responses = ["Shakespeare", "B", "0"]
        response_idx = [0]

        def make_response(*args, **kwargs):
            resp = MockResponse(responses[response_idx[0]])
            response_idx[0] = min(response_idx[0] + 1, len(responses) - 1)
            return MockContextManager(resp)

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(post=make_response)
            all_instances = asyncio.run(run_generation(all_instances, api_config))

        # Verify all instances got responses
        assert all(inst.response is not None for inst in all_instances)

        # Compute metrics for each task separately
        triviaqa_metrics = compute_metrics([all_instances[0]], triviaqa_config)
        openbookqa_metrics = compute_metrics([all_instances[1]], openbookqa_config)
        piqa_metrics = compute_metrics([all_instances[2]], piqa_config)

        # All tasks should have computed at least one metric
        assert len(triviaqa_metrics) > 0, "TriviaQA should have metrics"
        assert len(openbookqa_metrics) > 0, "OpenBookQA should have metrics"
        assert len(piqa_metrics) > 0, "PIQA should have metrics"

        # Report metrics
        print(f"\nTriviaQA metrics: {triviaqa_metrics}")
        print(f"OpenBookQA metrics: {openbookqa_metrics}")
        print(f"PIQA metrics: {piqa_metrics}")


class TestTaskConfigLoading:
    """Test that task configs load correctly."""

    @pytest.mark.parametrize(
        "task_path",
        [
            TASKS_DIR / "triviaqa" / "default.yaml",
            TASKS_DIR / "openbookqa" / "openbookqa.yaml",
            TASKS_DIR / "piqa" / "piqa.yaml",
        ],
    )
    def test_config_loads_successfully(self, task_path):
        """Verify each task config loads without errors."""
        from tinyeval import TaskConfig

        config = TaskConfig.from_yaml(task_path)
        assert config.task is not None
        assert config.dataset_path is not None
        assert config.doc_to_text is not None
