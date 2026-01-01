# tinyeval

Tiny Eval - A minimal harness for evaluating LLMs via OpenAI-compatible APIs.

> **Note:** This tool is designed for **comparing relative accuracy between inference frameworks** (e.g., vLLM vs TGI vs Ollama running the same model). It is not intended for absolute benchmark evaluations or leaderboard submissions. Use it to verify that different serving backends produce consistent results.

## Features

- **API-only** - works with any OpenAI-compatible endpoint (vLLM, TGI, Ollama, etc.)
- **Async HTTP** - concurrent requests with configurable parallelism
- **Two built-in tasks** - GSM8K (text) and ChartQA (multimodal)
- **Minimal dependencies** - just `datasets`, `aiohttp`, `pillow`

## Installation

```bash
pip install -e .
```

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
from core import APIConfig
from tasks.gsm8k import eval_gsm8k
from tasks.chartqa import eval_chartqa

# Configure API endpoint
config = APIConfig(
    url="http://localhost:8000/v1/chat/completions",
    model="gpt-4",
    num_concurrent=8
)

# Run GSM8K evaluation (text)
results = asyncio.run(eval_gsm8k(config, limit=100))
print(f"GSM8K: {results['metrics']}")

# Run ChartQA evaluation (multimodal)
results = asyncio.run(eval_chartqa(config, limit=100))
print(f"ChartQA: {results['metrics']}")
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
