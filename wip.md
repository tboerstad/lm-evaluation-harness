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
- `lm_eval/api/registry.py` - Model/task registry
- `lm_eval/evaluator.py` - Main evaluation logic (no multi-GPU)
- `lm_eval/loggers/utils.py` - Logging utilities
- `pyproject.toml` - Minimal dependencies

### Lines of Code Removed
- ~8,000 lines from model backends
- Significant reduction in evaluator.py (removed distributed code)

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
```

## Rollback
If something is broken, the original code is available in git history before the refactor commits.

## Known Limitations
- No local model inference (by design)
- No multi-GPU support (by design)
- `datasets` library still required for loading benchmarks
- Some task-specific utils still have torch/transformers imports (in `lm_eval/tasks/`)
