# tinyeval

Tiny Eval - A minimal harness for evaluating LLMs via OpenAI-compatible APIs.

## Features

- **Single-file implementation** (~400 lines) - easy to understand and modify
- **API-only** - works with any OpenAI-compatible endpoint (vLLM, TGI, Ollama, etc.)
- **Async HTTP** - concurrent requests with configurable parallelism
- **Two built-in tasks** - GSM8K (text) and ChartQA (multimodal)
- **Minimal dependencies** - just `datasets`, `aiohttp`, `pillow`

## Installation

```bash
pip install -e .
```

Or just copy `tinyeval.py` - it's a single file.

## Usage

### Command Line

```bash
# Evaluate GSM8K math benchmark (text)
python tinyeval.py \
    --tasks gsm8k_llama \
    --model gpt-4 \
    --base_url http://localhost:8000/v1/chat/completions \
    --limit 100

# Evaluate ChartQA (multimodal)
python tinyeval.py \
    --tasks chartqa \
    --model gpt-4-vision \
    --base_url http://localhost:8000/v1/chat/completions \
    --num_concurrent 4

# Both tasks
python tinyeval.py \
    --tasks gsm8k_llama,chartqa \
    --model llama-3 \
    --base_url http://localhost:8000/v1/chat/completions \
    --output results.json
```

### Python API

```python
import asyncio
from tinyeval import APIConfig, TASKS, build_instances, run_generation, compute_metrics, load_docs

# Get built-in task config
config = TASKS["gsm8k_llama"]

# Load dataset
docs = load_docs(config.dataset_path, config.dataset_name, limit=100)

# Build evaluation instances
instances = build_instances(config, docs)

# Configure API
api_config = APIConfig(
    base_url="http://localhost:8000/v1/chat/completions",
    model="gpt-4",
    num_concurrent=8
)

# Run evaluation
instances = asyncio.run(run_generation(instances, api_config, config))

# Compute metrics
metrics = compute_metrics(instances, config)
print(f"Results: {metrics}")
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tasks` | Comma-separated task names (gsm8k_llama, chartqa) | Required |
| `--model` | Model name for API requests | Required |
| `--base_url` | Chat completions API endpoint | Required |
| `--api_key` | API key (if needed) | "" |
| `--limit` | Limit number of samples | None |
| `--num_concurrent` | Concurrent HTTP requests | 8 |
| `--seed` | Random seed | 42 |
| `--output` | Output JSON file | None |

## Supported Tasks

| Task | Type | Dataset | Description |
|------|------|---------|-------------|
| `gsm8k_llama` | Text | gsm8k | Grade school math with chain-of-thought (8-shot) |
| `chartqa` | Multimodal | HuggingFaceM4/ChartQA | Chart question answering with images |

## Metrics

Built-in metrics:
- `exact_match` - Exact string match (with normalization)
- `relaxed_accuracy` - ChartQA metric with 5% numeric tolerance

## Output Format

```json
{
  "results": {
    "gsm8k_llama": {
      "metrics": {"exact_match": 0.85, "relaxed_accuracy": 0.87},
      "num_samples": 100,
      "time_seconds": 45.2
    }
  },
  "config": {
    "model": "gpt-4",
    "limit": 100
  },
  "total_time_seconds": 45.2
}
```
