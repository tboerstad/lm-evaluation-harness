# liteval

Lightweight LLM evaluation via OpenAI-compatible APIs.

## Features

- **Minimal**: Single-file implementation (~700 lines)
- **Async-first**: Concurrent HTTP requests with configurable limits
- **Flexible**: YAML task configs with Jinja2 templates
- **Multimodal**: Text and image support for vision models
- **Simple metrics**: exact_match, relaxed_accuracy, anywhere_accuracy

## Installation

```bash
pip install liteval

# For image/multimodal support:
pip install liteval[images]
```

## Quick Start

### CLI Usage

```bash
# Evaluate a model on GSM8K
liteval --tasks gsm8k --model gpt-4 --base_url https://api.openai.com/v1/chat/completions --api_key $OPENAI_API_KEY

# With sample limit and custom concurrency
liteval --tasks gsm8k --model local-model --base_url http://localhost:8000/v1/chat/completions --limit 100 --num_concurrent 16
```

### Python API

```python
import asyncio
from pathlib import Path
from liteval import APIConfig, TaskConfig, evaluate_task

async def main():
    config = APIConfig(
        base_url="http://localhost:8000/v1/chat/completions",
        model="my-model",
        num_concurrent=8,
    )

    result, eval_time = await evaluate_task(
        task_path=Path("tasks/gsm8k.yaml"),
        api_config=config,
        limit=100,
    )

    print(f"Task: {result.task}")
    print(f"Metrics: {result.metrics}")
    print(f"Samples: {result.num_samples}")
    print(f"Time: {eval_time:.2f}s")

asyncio.run(main())
```

## Task Configuration

Tasks are defined in YAML files:

```yaml
task: my_task
dataset_path: dataset/name
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: answer
num_fewshot: 5
generation_kwargs:
  max_tokens: 256
  temperature: 0
  until: ["\n\n"]
metric_list:
  - metric: exact_match
    ignore_case: true
```

### Multimodal Tasks

```yaml
task: chartqa
dataset_path: HuggingFaceM4/ChartQA
doc_to_text: "<image>\n{{query}}"
doc_to_target: label
doc_to_image: image
metric_list:
  - metric: relaxed_accuracy
```

## Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | Exact string match (supports ignore_case, ignore_punctuation) |
| `relaxed_accuracy` | ChartQA-style: exact match or 5% numeric tolerance |
| `anywhere_accuracy` | Reference appears anywhere in response |

## API Configuration

```python
from liteval import APIConfig

config = APIConfig(
    base_url="http://localhost:8000/v1/chat/completions",  # Chat completions endpoint
    model="model-name",           # Model name for API
    api_key="",                   # Bearer token (optional for local)
    max_tokens=512,               # Default max generation tokens
    temperature=0.0,              # Sampling temperature
    seed=1234,                    # Random seed
    num_concurrent=8,             # Max concurrent requests
    timeout=300,                  # Request timeout (seconds)
    max_retries=3,                # Retry attempts
)
```

## CLI Options

```
--tasks          Comma-separated task names or YAML paths
--model          Model name for API requests
--base_url       Chat completions endpoint URL
--api_key        API key (optional)
--num_fewshot    Number of few-shot examples (overrides task config)
--limit          Limit number of samples
--num_concurrent Concurrent requests (default: 8)
--seed           Random seed (default: 42)
--tasks_dir      Directory to search for task YAMLs
--output         Save results to JSON file
```

## Included Tasks

liteval includes 200+ benchmark task configurations from the original lm-evaluation-harness:

- **Math**: gsm8k, math, asdiv, svamp
- **Reasoning**: arc, hellaswag, winogrande, piqa
- **Knowledge**: mmlu, truthfulqa, triviaqa
- **Code**: humaneval, mbpp
- **Multimodal**: chartqa, docvqa, textvqa
- And many more...

Browse available tasks in the `tasks/` directory.

## License

MIT
