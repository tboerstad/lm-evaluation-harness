# liteeval

Lightweight LM Evaluation - A minimal harness for evaluating LLMs via OpenAI-compatible APIs.

## Features

- **Single-file implementation** (~600 lines) - easy to understand and modify
- **API-only** - works with any OpenAI-compatible endpoint (vLLM, TGI, Ollama, etc.)
- **Async HTTP** - concurrent requests with configurable parallelism
- **Multimodal** - supports vision tasks with images (ChartQA, etc.)
- **YAML tasks** - 200+ task configurations included
- **Minimal dependencies** - just `datasets`, `aiohttp`, `jinja2`, `pyyaml`, `pillow`

## Installation

```bash
pip install -e .
```

Or just copy `liteeval.py` - it's a single file with no package dependencies.

## Usage

### Command Line

```bash
# Evaluate GSM8K math benchmark
python liteeval.py \
    --tasks gsm8k \
    --model gpt-4 \
    --base_url http://localhost:8000/v1/chat/completions \
    --limit 100

# Evaluate multimodal ChartQA
python liteeval.py \
    --tasks chartqa \
    --model gpt-4-vision \
    --base_url http://localhost:8000/v1/chat/completions \
    --num_concurrent 4

# Multiple tasks with few-shot examples
python liteeval.py \
    --tasks gsm8k,hellaswag \
    --model llama-3 \
    --base_url http://localhost:8000/v1/chat/completions \
    --num_fewshot 5 \
    --output results.json
```

### Python API

```python
import asyncio
from pathlib import Path
from liteeval import APIConfig, TaskConfig, build_instances, run_generation, compute_metrics, load_all_docs

# Load task configuration
config = TaskConfig.from_yaml(Path("tasks/gsm8k/gsm8k.yaml"))

# Load dataset
docs = load_all_docs(config.dataset_path, config.dataset_name, limit=100)

# Build evaluation instances
instances = build_instances(config, docs)

# Configure API
api_config = APIConfig(
    base_url="http://localhost:8000/v1/chat/completions",
    model="gpt-4",
    num_concurrent=8
)

# Run evaluation
instances = asyncio.run(run_generation(instances, api_config))

# Compute metrics
metrics = compute_metrics(instances, config)
print(f"Results: {metrics}")
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tasks` | Comma-separated task names | Required |
| `--model` | Model name for API requests | Required |
| `--base_url` | Chat completions API endpoint | Required |
| `--api_key` | API key (if needed) | "" |
| `--num_fewshot` | Number of few-shot examples | None |
| `--limit` | Limit number of samples | None |
| `--num_concurrent` | Concurrent HTTP requests | 8 |
| `--seed` | Random seed | 42 |
| `--tasks_dir` | Directory with task YAML files | "tasks" |
| `--output` | Output JSON file | None |

## Supported Tasks

All YAML task configurations are included. Common tasks:

- **Math**: `gsm8k`, `math`, `minerva_math`
- **Reasoning**: `arc_challenge`, `hellaswag`, `winogrande`
- **Knowledge**: `mmlu`, `triviaqa`, `naturalqs`
- **Multimodal**: `chartqa`, `docvqa`, `textvqa`
- **Code**: `humaneval`, `mbpp`

See the `tasks/` directory for all available configurations.

## Metrics

Built-in metrics:
- `exact_match` - Exact string match (with optional case/punctuation normalization)
- `relaxed_accuracy` - ChartQA metric with 5% numeric tolerance
- `anywhere_accuracy` - Substring match anywhere in response

## Output Format

```json
{
  "results": {
    "gsm8k": {
      "metrics": {"exact_match": 0.85},
      "num_samples": 100,
      "evaluation_time_seconds": "45.2"
    }
  },
  "config": {
    "model": "gpt-4",
    "num_fewshot": 5,
    "limit": 100
  },
  "total_evaluation_time_seconds": "45.2"
}
```

## License

MIT
