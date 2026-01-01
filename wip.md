# Work in Progress: OpenAI API-Only Refactor

## Goal
Strip down lm-evaluation-harness to only support OpenAI-compatible REST APIs, removing all local model backends and heavy dependencies (torch, transformers, etc.).

## What Was Done

### Phase 1: Remove Non-OpenAI Model Backends
Deleted 18 model files from `lm_eval/models/`:
- `anthropic_llms.py`
- `huggingface.py`
- `vllm_causallms.py`, `vllm_vlms.py`
- `sglang_causallms.py`, `sglang_generate_API.py`
- `mamba_lm.py`, `nemo_lm.py`
- `gguf.py`, `dummy.py`
- `hf_vlms.py`, `hf_audiolm.py`, `hf_steered.py`
- `ibm_watsonx_ai.py`, `textsynth.py`
- `neuron_optimum.py`, `optimum_lm.py`, `optimum_ipex.py`

### Phase 2: Update Model Imports
Updated `lm_eval/models/__init__.py` to only import:
- `api_models`
- `openai_completions`

### Phase 3: Remove Torch/Transformers Dependencies
Made torch and transformers optional throughout the codebase:

1. **`lm_eval/models/utils.py`** - Removed torch imports, kept only utility functions needed for API models (Collator, chunks, handle_stop_sequences, etc.)

2. **`lm_eval/models/api_models.py`** - Removed tokenizer backend options, simplified to API-only mode (no local tokenizer)

3. **`lm_eval/api/model.py`** - Made transformers import optional in `_encode_pair` method

4. **`lm_eval/api/registry.py`** - Made `evaluate` (HF) import optional

5. **`lm_eval/evaluator.py`** - Removed torch import and all multi-GPU/distributed code

6. **`lm_eval/loggers/utils.py`** - Made torch and transformers imports optional

### Phase 4: Update Dependencies
Updated `pyproject.toml`:
- Removed: `torch`, `transformers`, `accelerate`, `peft`, `tiktoken`
- Removed: `vllm`, `mamba_ssm`, `optimum`, etc. from optional deps
- Kept: `datasets` (still needed for loading benchmark data)
- Kept: HTTP deps (`requests`, `aiohttp`, `tenacity`, `tqdm`)

### Phase 5: Consolidate Model Classes
Simplified `lm_eval/models/openai_completions.py`:
- Merged `OpenAICompletionsAPI` into `LocalCompletionsAPI`
- Merged `OpenAIChatCompletion` into `LocalChatCompletion`
- Both `local-*` and `openai-*` names now point to the same classes (backwards compatible)

### Phase 6: Remove Optional Dependencies and Integrations
Removed all optional dependencies and their related code:

1. **Optional dependency groups from `pyproject.toml`:**
   - `ifeval`: langdetect, immutabledict, nltk
   - `math`: sympy, antlr4-python3-runtime, math_verify
   - `multilingual`: nagisa, jieba, pycountry

2. **hf_transfer integration:**
   - Removed from `lm_eval/__init__.py`

3. **HuggingFace evaluate library integration:**
   - Removed `HAS_HF_EVALUATE` and fallback logic from `lm_eval/api/registry.py`
   - Simplified `get_metric()` function to only use local registry
   - Removed `hf_evaluate_metric` parameter handling from `lm_eval/api/task.py`

4. **Weights & Biases (wandb) integration:**
   - Deleted `lm_eval/loggers/wandb_logger.py`
   - Removed `WandbLogger` from `lm_eval/loggers/__init__.py`
   - Removed `--wandb_args` and `--wandb_config_args` CLI arguments
   - Removed `wandb_args` and `wandb_config_args` from `EvaluatorConfig`

5. **Zeno visualization:**
   - Deleted `scripts/zeno_visualize.py`
   - Deleted `tests/scripts/test_zeno_visualize.py`

6. **Example files removed:**
   - `examples/visualize-wandb.ipynb`
   - `examples/visualize-zeno.ipynb`
   - `examples/transformer-lens.py` (used removed HF backend)

7. **Documentation updates:**
   - Removed wandb_args from `docs/config_files.md` and `docs/interface.md`
   - Removed "Visualizing Results" section from `README.md`
   - Simplified Optional Extras section in `README.md`

### Phase 7: Add CLI HTTP Request Tests
Added comprehensive tests for CLI HTTP request handling (`tests/test_cli_http.py`):

1. **Sync HTTP request tests:**
   - Payload format validation (model, max_tokens, temperature)
   - Stop sequences correctly passed in payload
   - Multiple requests count verification
   - URL and headers validation

2. **Async HTTP request tests:**
   - Batched request structure verification
   - Request count matching input (N requests → N HTTP calls)
   - TCPConnector concurrency limit (num_concurrent)

3. **Task-style request tests:**
   - GSM8K-style prompts with stop sequences (`["Question:", "</s>", "<|im_end|>"]`)
   - ChartQA-style generation kwargs (max_gen_toks: 512)

4. **Request payload validation:**
   - Seed (default 1234) included in payload
   - Custom model names passed correctly
   - Batch size handling

## Current State

### Available Models
| Name | Alias | Description |
|------|-------|-------------|
| `local-completions` | `openai-completions` | OpenAI-compatible /v1/completions API |
| `local-chat-completions` | `openai-chat-completions` | OpenAI-compatible /v1/chat/completions API |

### Key Files Modified
- `lm_eval/models/__init__.py` - Simplified imports
- `lm_eval/models/openai_completions.py` - 2 model classes (~190 lines)
- `lm_eval/models/api_models.py` - Base TemplateAPI class
- `lm_eval/models/utils.py` - Utility functions (no torch)
- `lm_eval/api/model.py` - Base LM class
- `lm_eval/api/registry.py` - Model/task registry (no HF evaluate fallback)
- `lm_eval/api/task.py` - Task configuration (removed hf_evaluate handling)
- `lm_eval/evaluator.py` - Main evaluation logic (no multi-GPU)
- `lm_eval/loggers/__init__.py` - Only exports EvaluationTracker
- `lm_eval/loggers/utils.py` - Logging utilities
- `lm_eval/_cli/run.py` - CLI (removed wandb args)
- `lm_eval/config/evaluate_config.py` - Config (removed wandb fields)
- `pyproject.toml` - Minimal dependencies (only dev/testing extras)
- `tests/test_cli_http.py` - CLI HTTP request tests (12 tests)

### Lines of Code Removed
- ~8,000 lines from model backends
- Significant reduction in evaluator.py (removed distributed code)
- ~1,200 lines from optional integrations (wandb, zeno, examples)

## How to Test
```bash
# Test imports work
python -c "
import lm_eval.models
from lm_eval.api.registry import MODEL_REGISTRY
print('Models:', list(MODEL_REGISTRY.keys()))
"

# Test CLI
python -m lm_eval --help

# Run CLI HTTP tests (12 tests covering sync/async HTTP requests)
python -m pytest tests/test_cli_http.py -v
```

## Rollback
If something is broken, the original code is available in git history before the refactor commits.

## Known Limitations
- No local model inference (by design)
- No multi-GPU support (by design)
- `datasets` library still required for loading benchmarks
- Some task-specific utils still have torch/transformers imports (in `lm_eval/tasks/`)
- No W&B or Zeno result visualization (removed)
- No HuggingFace evaluate metrics fallback (removed)
- Task-specific optional dependencies removed (ifeval, math, multilingual) - tasks using these may have reduced functionality
