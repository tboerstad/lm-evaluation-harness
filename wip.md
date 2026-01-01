# tinyeval - Tiny Eval

## Overview

tinyeval is a minimal, single-file evaluation harness for LLMs via OpenAI-compatible APIs. It was created by stripping down lm-evaluation-harness to remove all local model backends and heavy dependencies (torch, transformers, etc.).

## Final Architecture

```
tinyeval/
├── tinyeval.py       # Single-file harness (~600 lines)
├── tasks/            # 200+ YAML task configurations
├── tests/            # 18 passing tests
│   ├── test_tinyeval.py
│   └── test_chartqa_integration.py
├── pyproject.toml    # Minimal dependencies
└── README.md
```

## Core Components

### tinyeval.py (~600 lines)

**Dataclasses:**
- `TaskConfig` - YAML task configuration (dataset, prompts, metrics, generation settings)
- `Instance` - Single evaluation instance (doc, prompt, target, images, response)
- `EvalResult` - Evaluation results (task name, metrics, sample count)
- `APIConfig` - OpenAI-compatible API configuration

**Key Functions:**
- `render_template()` - Jinja2 template rendering
- `build_instances()` - Convert documents to evaluation instances
- `build_fewshot_context()` - Format few-shot examples
- `run_generation()` - Async HTTP requests with concurrency control
- `compute_metrics()` - Calculate metrics over instances

**Metrics:**
- `exact_match()` - Exact string match (with normalization options)
- `relaxed_accuracy()` - ChartQA metric with 5% numeric tolerance
- `anywhere_accuracy()` - Substring match anywhere in response

**Multimodal Support:**
- `encode_image_to_base64()` - PIL Image to base64
- `build_multimodal_message()` - Vision API message format
- `get_images_from_doc()` - Extract images from documents

### Dependencies (Minimal)

```
datasets>=2.16.0    # HuggingFace datasets for benchmarks
aiohttp             # Async HTTP client
requests            # Sync HTTP client
jinja2              # Template rendering
pyyaml              # YAML config parsing
pillow              # Image handling
```

## What Was Removed

### Phase 1-8: Initial Cleanup (from lm-evaluation-harness)
- 18 model backend files (HuggingFace, vLLM, SGLang, Anthropic, etc.)
- torch/transformers dependencies
- Request caching system
- Weights & Biases integration
- Zeno visualization
- HuggingFace evaluate library fallback
- Multi-GPU/distributed code

### Phase 9-13: Mini Implementation
- Created standalone `lm_eval_mini.py`
- Simplified to generation-only (no logprobs/multiple_choice)
- Removed split field complexity
- Added comprehensive tests

### Phase 14: Rebrand to liteeval

### Phase 15: Rebrand to tinyeval
- Renamed `lm_eval_mini.py` → `tinyeval.py`
- Moved `lm_eval/tasks/` → `tasks/`
- Deleted entire `lm_eval/` package (~12K lines)
- Removed `docs/`, `examples/`, `scripts/`, `templates/`
- Removed `.github/`, `CITATION.bib`, `CODEOWNERS`, etc.
- Updated `pyproject.toml` with new name and minimal deps
- Rewrote `README.md` for tinyeval

## Usage

### Command Line

```bash
python tinyeval.py \
    --tasks gsm8k \
    --model gpt-4 \
    --base_url http://localhost:8000/v1/chat/completions \
    --limit 100
```

### Python API

```python
import asyncio
from pathlib import Path
from tinyeval import APIConfig, TaskConfig, build_instances, run_generation, compute_metrics, load_all_docs

config = TaskConfig.from_yaml(Path("tasks/gsm8k/gsm8k.yaml"))
docs = load_all_docs(config.dataset_path, config.dataset_name, limit=100)
instances = build_instances(config, docs)

api_config = APIConfig(
    base_url="http://localhost:8000/v1/chat/completions",
    model="gpt-4",
    num_concurrent=8
)

instances = asyncio.run(run_generation(instances, api_config))
metrics = compute_metrics(instances, config)
```

## Tests

18 tests covering:
- HTTP request formation (sync and async)
- Image encoding and multimodal messages
- Metrics (exact_match, relaxed_accuracy, anywhere_accuracy)
- Task configuration loading
- ChartQA end-to-end pipeline

Run tests:
```bash
python -m pytest tests/ -v
```

## Design Decisions

1. **Single-file implementation** - Easy to understand, copy, and modify
2. **API-only** - No local model inference, works with any OpenAI-compatible endpoint
3. **No train/val split distinction** - Loads all data, for inference framework comparisons
4. **Async HTTP** - Concurrent requests with configurable parallelism
5. **YAML tasks preserved** - All 200+ task configs from lm-evaluation-harness work

## Known Limitations

- No local model inference (by design)
- No multi-GPU support (by design)
- No request/response caching (removed)
- No train/val/test split distinction (by design)
- Some task-specific Python utils in `tasks/` may reference removed `lm_eval` imports
