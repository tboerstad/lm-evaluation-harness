"""Test running a single question from multiple tasks.

This test verifies that the evaluation pipeline correctly:
1. Loads task configurations for sciq, commonsense_qa, and asdiv
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
    """Test running a single question from sciq, commonsense_qa, and asdiv."""

    @pytest.fixture
    def sciq_config(self):
        """Load sciq task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "sciq" / "sciq.yaml")

    @pytest.fixture
    def commonsense_qa_config(self):
        """Load commonsense_qa task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "commonsense_qa" / "default.yaml")

    @pytest.fixture
    def asdiv_config(self):
        """Load asdiv task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "asdiv" / "default.yaml")

    def test_sciq_single_question(self, sciq_config):
        """Test running a single SciQ science question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock SciQ document
        doc = {
            "support": "The mitochondria is the powerhouse of the cell, responsible for producing ATP.",
            "question": "What organelle produces ATP in cells?",
            "distractor1": "Nucleus",
            "distractor2": "Ribosome",
            "distractor3": "Golgi apparatus",
            "correct_answer": "Mitochondria",
        }

        instances = build_instances(sciq_config, [doc])
        assert len(instances) == 1
        assert "ATP" in instances[0].prompt
        assert instances[0].target is not None

        # Mock API response with correct answer
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("Mitochondria")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        assert "Mitochondria" in instances[0].response

        metrics = compute_metrics(instances, sciq_config)
        assert len(metrics) > 0

    def test_commonsense_qa_single_question(self, commonsense_qa_config):
        """Test running a single CommonsenseQA question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock CommonsenseQA document
        doc = {
            "question": "Where would you put a plant?",
            "choices": {
                "text": ["garden", "freezer", "oven", "underwater", "space"],
                "label": ["A", "B", "C", "D", "E"],
            },
            "answerKey": "A",
        }

        instances = build_instances(commonsense_qa_config, [doc])
        assert len(instances) == 1
        assert "plant" in instances[0].prompt

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("A")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        metrics = compute_metrics(instances, commonsense_qa_config)
        assert len(metrics) > 0

    def test_asdiv_single_question(self, asdiv_config):
        """Test running a single ASDiv math word problem."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock ASDiv document
        doc = {
            "body": "Tom has 5 apples. He buys 3 more apples from the store.",
            "question": "How many apples does Tom have now?",
            "answer": "8 (apples)",
        }

        instances = build_instances(asdiv_config, [doc])
        assert len(instances) == 1
        assert "apples" in instances[0].prompt

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("8")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        metrics = compute_metrics(instances, asdiv_config)
        assert len(metrics) > 0

    def test_all_three_tasks_together(self, sciq_config, commonsense_qa_config, asdiv_config):
        """Test running all three tasks together in a single pipeline."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create mock documents for each task
        sciq_doc = {
            "support": "Water freezes at 0 degrees Celsius under normal conditions.",
            "question": "At what temperature does water freeze?",
            "distractor1": "100°C",
            "distractor2": "50°C",
            "distractor3": "-100°C",
            "correct_answer": "0°C",
        }
        commonsense_qa_doc = {
            "question": "What do you use to cut paper?",
            "choices": {
                "text": ["scissors", "hammer", "spoon", "pillow", "blanket"],
                "label": ["A", "B", "C", "D", "E"],
            },
            "answerKey": "A",
        }
        asdiv_doc = {
            "body": "Sarah has 10 candies. She gives 4 candies to her friend.",
            "question": "How many candies does Sarah have left?",
            "answer": "6 (candies)",
        }

        # Build instances for each task
        sciq_instances = build_instances(sciq_config, [sciq_doc])
        commonsense_qa_instances = build_instances(commonsense_qa_config, [commonsense_qa_doc])
        asdiv_instances = build_instances(asdiv_config, [asdiv_doc])

        all_instances = sciq_instances + commonsense_qa_instances + asdiv_instances
        assert len(all_instances) == 3

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )

        responses = ["0°C", "A", "6"]
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
        sciq_metrics = compute_metrics([all_instances[0]], sciq_config)
        commonsense_qa_metrics = compute_metrics([all_instances[1]], commonsense_qa_config)
        asdiv_metrics = compute_metrics([all_instances[2]], asdiv_config)

        # All tasks should have computed at least one metric
        assert len(sciq_metrics) > 0, "SciQ should have metrics"
        assert len(commonsense_qa_metrics) > 0, "CommonsenseQA should have metrics"
        assert len(asdiv_metrics) > 0, "ASDiv should have metrics"

        # Report metrics
        print(f"\nSciQ metrics: {sciq_metrics}")
        print(f"CommonsenseQA metrics: {commonsense_qa_metrics}")
        print(f"ASDiv metrics: {asdiv_metrics}")


class TestTaskConfigLoading:
    """Test that task configs load correctly."""

    @pytest.mark.parametrize(
        "task_path",
        [
            TASKS_DIR / "sciq" / "sciq.yaml",
            TASKS_DIR / "commonsense_qa" / "default.yaml",
            TASKS_DIR / "asdiv" / "default.yaml",
        ],
    )
    def test_config_loads_successfully(self, task_path):
        """Verify each task config loads without errors."""
        from tinyeval import TaskConfig

        config = TaskConfig.from_yaml(task_path)
        assert config.task is not None
        assert config.dataset_path is not None
        assert config.doc_to_text is not None
