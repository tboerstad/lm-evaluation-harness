"""Test running a single question from multiple tasks.

This test verifies that the evaluation pipeline correctly:
1. Loads task configurations for gsm8k, arc_easy, and hellaswag
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
    """Test running a single question from gsm8k, arc_easy, and hellaswag."""

    @pytest.fixture
    def gsm8k_config(self):
        """Load gsm8k task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "gsm8k" / "gsm8k.yaml")

    @pytest.fixture
    def arc_easy_config(self):
        """Load arc_easy task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "arc" / "arc_easy.yaml")

    @pytest.fixture
    def hellaswag_config(self):
        """Load hellaswag task config."""
        from tinyeval import TaskConfig

        return TaskConfig.from_yaml(TASKS_DIR / "hellaswag" / "hellaswag.yaml")

    def test_gsm8k_single_question(self, gsm8k_config):
        """Test running a single GSM8K math question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock GSM8K document
        doc = {
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake muffins. How many does she have left?",
            "answer": "Janet has 16 - 3 - 4 = <<16-3-4=9>>9 eggs left. #### 9",
        }

        instances = build_instances(gsm8k_config, [doc])
        assert len(instances) == 1
        assert "Janet's ducks" in instances[0].prompt
        assert instances[0].target is not None

        # Mock API response with correct answer format
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("Let me solve this step by step.\n16 - 3 - 4 = 9\n#### 9")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        assert "9" in instances[0].response

        metrics = compute_metrics(instances, gsm8k_config)
        assert "exact_match" in metrics

    def test_arc_easy_single_question(self, arc_easy_config):
        """Test running a single ARC-Easy question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock ARC-Easy document
        doc = {
            "question": "Which of these is a nonrenewable resource?",
            "choices": {"text": ["Coal", "Trees", "Water", "Sunlight"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        }

        instances = build_instances(arc_easy_config, [doc])
        assert len(instances) == 1
        assert "nonrenewable" in instances[0].prompt

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
        metrics = compute_metrics(instances, arc_easy_config)
        # ARC uses acc metric but we compute exact_match by default
        assert len(metrics) > 0

    def test_hellaswag_single_question(self, hellaswag_config):
        """Test running a single HellaSwag question."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create a mock HellaSwag document (processed format)
        doc = {
            "query": "A woman is outside with a bucket and a dog. The dog is running around trying to catch water from the bucket. The woman",
            "choices": [
                "rinses the bucket and puts it down.",
                "splashes water on the dog.",
                "picks up the dog.",
                "throws the bucket at the dog.",
            ],
            "label": "1",  # Second choice is correct
        }

        instances = build_instances(hellaswag_config, [doc])
        assert len(instances) == 1
        assert "bucket" in instances[0].prompt

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )
        mock_response = MockResponse("1")

        with patch("tinyeval.aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value = AsyncMock(
                post=lambda *a, **k: MockContextManager(mock_response)
            )
            instances = asyncio.run(run_generation(instances, api_config))

        assert instances[0].response is not None
        metrics = compute_metrics(instances, hellaswag_config)
        assert len(metrics) > 0

    def test_all_three_tasks_together(self, gsm8k_config, arc_easy_config, hellaswag_config):
        """Test running all three tasks together in a single pipeline."""
        from tinyeval import APIConfig, build_instances, compute_metrics, run_generation

        # Create mock documents for each task
        gsm8k_doc = {
            "question": "If there are 3 apples and you take 2, how many do you have?",
            "answer": "You have 2 apples because you took them. #### 2",
        }
        arc_doc = {
            "question": "What is the chemical formula for water?",
            "choices": {"text": ["H2O", "CO2", "O2", "N2"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        }
        hellaswag_doc = {
            "query": "A chef is in the kitchen preparing a meal. The chef",
            "choices": ["burns the food.", "adds salt.", "leaves.", "dances."],
            "label": "1",
        }

        # Build instances for each task
        gsm8k_instances = build_instances(gsm8k_config, [gsm8k_doc])
        arc_instances = build_instances(arc_easy_config, [arc_doc])
        hellaswag_instances = build_instances(hellaswag_config, [hellaswag_doc])

        all_instances = gsm8k_instances + arc_instances + hellaswag_instances
        assert len(all_instances) == 3

        # Mock API response
        api_config = APIConfig(
            base_url="http://mock/v1/chat/completions", model="mock", api_key="test"
        )

        responses = ["#### 2", "A", "1"]
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
        gsm8k_metrics = compute_metrics([all_instances[0]], gsm8k_config)
        arc_metrics = compute_metrics([all_instances[1]], arc_easy_config)
        hellaswag_metrics = compute_metrics([all_instances[2]], hellaswag_config)

        # All tasks should have computed at least one metric
        assert len(gsm8k_metrics) > 0, "GSM8K should have metrics"
        assert len(arc_metrics) > 0, "ARC-Easy should have metrics"
        assert len(hellaswag_metrics) > 0, "HellaSwag should have metrics"

        # Report metrics
        print(f"\nGSM8K metrics: {gsm8k_metrics}")
        print(f"ARC-Easy metrics: {arc_metrics}")
        print(f"HellaSwag metrics: {hellaswag_metrics}")


class TestTaskConfigLoading:
    """Test that task configs load correctly."""

    @pytest.mark.parametrize(
        "task_path",
        [
            TASKS_DIR / "gsm8k" / "gsm8k.yaml",
            TASKS_DIR / "arc" / "arc_easy.yaml",
            TASKS_DIR / "hellaswag" / "hellaswag.yaml",
        ],
    )
    def test_config_loads_successfully(self, task_path):
        """Verify each task config loads without errors."""
        from tinyeval import TaskConfig

        config = TaskConfig.from_yaml(task_path)
        assert config.task is not None
        assert config.dataset_path is not None
        assert config.doc_to_text is not None
