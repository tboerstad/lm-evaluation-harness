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

### Phase 8: Remove Request Caching
Removed all caching functionality from the codebase:

1. **Deleted files:**
   - `lm_eval/caching/` directory (`__init__.py`, `cache.py`)
   - `tests/test_requests_caching.py`
   - `scripts/requests_caching.py`

2. **Removed from `lm_eval/api/model.py`:**
   - `CachingLM` class (SQLite-based response caching wrapper)
   - `CacheHook` class
   - `hash_args()` function
   - `sqlitedict` import

3. **Removed from `lm_eval/api/task.py`:**
   - `cache_requests` and `rewrite_requests_cache` parameters from `build_all_requests()`
   - Cache key generation and loading/saving logic
   - Import of `load_from_cache`, `save_to_cache`

4. **Removed from CLI (`lm_eval/_cli/run.py`):**
   - `--use_cache` / `-c` argument
   - `--cache_requests` argument (true/refresh/delete)
   - "caching and performance" argument group

5. **Removed from `lm_eval/_cli/utils.py`:**
   - `request_caching_arg_to_dict()` function

6. **Removed from `lm_eval/config/evaluate_config.py`:**
   - `use_cache` field
   - `cache_requests` field

7. **Removed from `lm_eval/evaluator.py`:**
   - `use_cache`, `cache_requests`, `rewrite_requests_cache`, `delete_requests_cache` parameters
   - `CachingLM` wrapper instantiation
   - `delete_cache()` call
   - `request_caching_arg_to_dict()` function

8. **Updated `pyproject.toml`:**
   - Removed `sqlitedict` dependency

9. **Updated `tests/test_cli_subcommands.py`:**
   - Removed `request_caching_arg_to_dict` import and tests

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
- ~570 lines from caching functionality

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

### Phase 9: Minimal Standalone Implementation

Created `lm_eval_mini.py` (~900 lines) - a standalone, minimal reimplementation that:

1. **Core Design:**
   - Single-file implementation with no external lm_eval dependencies
   - Supports generation-only tasks (no logprobs/multiple_choice)
   - Uses OpenAI-compatible chat completions API (required for vision models)
   - Async HTTP with aiohttp and semaphore-based concurrency control

2. **Supported Tasks:**
   - **GSM8K** - Math reasoning benchmark
   - **ChartQA** - Multimodal chart question answering (images + text)

3. **Key Components:**
   - `TaskConfig` - Dataclass for YAML task configuration
   - `Instance` - Request instance with document and metadata
   - `APIConfig` - API endpoint configuration
   - `LocalCompletionsAPI` - Main API client with sync/async support

4. **Multimodal/Image Support:**
   - `encode_image_to_base64()` - Encodes PIL images to base64
   - `build_multimodal_message()` - Builds vision API message format
   - `get_images_from_doc()` - Extracts images from documents
   - `TaskConfig.is_multimodal` - Property to detect image tasks

5. **Metrics:**
   - `exact_match()` - Exact string matching (case-insensitive option)
   - `relaxed_accuracy()` - ChartQA metric with 5% numeric tolerance
   - `anywhere_accuracy()` - Substring matching anywhere in response

6. **YAML Config Loading:**
   - `load_yaml_task_config()` - Parses YAML with jinja2 templates
   - Supports `doc_to_text`, `doc_to_target`, `doc_to_image` fields
   - Handles `generation_kwargs` (temperature, max_gen_toks, until)

7. **Files Created:**
   - `lm_eval_mini.py` - Main implementation (~900 lines)
   - `tests/test_lm_eval_mini.py` - Comprehensive tests (~330 lines)

8. **Test Coverage:**
   - HTTP request formation (sync and async)
   - Image/multimodal message building
   - Metrics (exact_match, relaxed_accuracy, anywhere_accuracy)
   - Task configuration loading
   - Payload structure validation

### Phase 10: ChartQA Integration Testing

Added comprehensive integration tests for ChartQA and fixed YAML loading:

1. **Fixed YAML `!function` tag handling** (`lm_eval_mini.py:64-78`):
   - ChartQA config uses `!function utils.exact_match` syntax
   - `yaml.safe_load()` doesn't support custom tags
   - Added `FunctionTagLoader` class extending `yaml.SafeLoader`
   - Custom constructor converts `!function` tags to strings

2. **Created `tests/test_chartqa_integration.py`** (~350 lines):
   - `TestChartQADatasetDownload` - Verifies HuggingFace dataset download
   - `TestChartQATaskConfig` - Validates YAML config loading
   - `TestChartQAInstanceBuilding` - Tests prompt/instance generation
   - `TestChartQAMultimodalMessages` - Verifies image encoding and vision API format
   - `TestChartQAEndToEnd` - Full pipeline test with mocked API
   - `TestChartQADatasetFullLoad` - Verifies full dataset load

3. **Test Coverage (18 new tests):**
   | Test | Description |
   |------|-------------|
   | `test_dataset_downloads_successfully` | ChartQA downloads from HuggingFace |
   | `test_dataset_has_required_fields` | Has image, query, label fields |
   | `test_image_field_is_pil_image` | Image field is PIL Image |
   | `test_query_field_is_string` | Query is non-empty string |
   | `test_label_field_is_list` | Label is list of answers |
   | `test_config_file_exists` | chartqa.yaml exists |
   | `test_config_loads_successfully` | TaskConfig parses YAML |
   | `test_config_has_multimodal_settings` | is_multimodal=True, doc_to_image set |
   | `test_config_has_doc_to_text_template` | Template has <image>, {{query}} |
   | `test_config_has_metrics` | metric_list is populated |
   | `test_build_instances_creates_correct_count` | N samples → N instances |
   | `test_instances_have_images` | Each instance has images extracted |
   | `test_instances_have_prompts` | Prompts contain rendered query |
   | `test_instances_have_targets` | Targets match label[0] |
   | `test_image_encodes_to_base64` | PIL image → base64 string |
   | `test_multimodal_message_built_correctly` | Vision API message format |
   | `test_full_evaluation_pipeline_with_mock_api` | End-to-end with mocked HTTP |
   | `test_full_dataset_load` | Non-streaming full load works |

4. **Test Results:**
   ```
   tests/test_chartqa_integration.py: 18 passed
   tests/test_lm_eval_mini.py: 18 passed
   Total: 36 tests passing
   ```

### Phase 11: Remove Split Fields Entirely

Simplified `lm_eval_mini.py` to ignore splits completely - loads all available data from the dataset. This is intentional - the mini harness is for relative comparisons between inference frameworks, not for train/val separation.

1. **Simplified `TaskConfig`:**
   - Removed: `test_split`, `training_split`, `fewshot_split`, `split`
   - No split configuration at all

2. **Added `load_all_docs()`:**
   - Loads all splits from dataset and concatenates them
   - Applies `[:limit]` slice if limit specified

3. **Updated `get_fewshot_examples()`:**
   - Now takes a `docs` list directly instead of dataset + config
   - Samples few-shot examples from loaded data
   - Added optional `exclude_indices` parameter

4. **Test updates:**
   - Updated seed reproducibility tests for new function signature
   - Added `test_fewshot_exclude_indices` test

### Phase 12: Add Dataclass Field Documentation

Added single-line documentation for all dataclass fields in `lm_eval_mini.py`:

1. **TaskConfig** - 12 fields documented:
   - `task`: Unique identifier for the task
   - `dataset_path`: HuggingFace dataset path or local directory
   - `dataset_name`: Dataset configuration/subset name
   - `num_fewshot`: Number of few-shot examples to prepend
   - `doc_to_text`: Jinja2 template for input prompt
   - `doc_to_target`: Template or field name for expected answer
   - `doc_to_image`: Field(s) containing images for multimodal tasks
   - `target_delimiter`: Separator between prompt and target in few-shot
   - `fewshot_delimiter`: Separator between few-shot examples
   - `generation_kwargs`: API generation params (temperature, max_tokens, until)
   - `metric_list`: Metrics to compute
   - `filter_list`: Post-processing filters for answer extraction

2. **Instance** - 7 fields documented:
   - `doc`: Original document from the dataset
   - `doc_id`: Index of the document in the dataset
   - `prompt`: Rendered prompt text with few-shot context
   - `target`: Expected answer(s) for evaluation
   - `images`: PIL Images or base64 strings for multimodal
   - `generation_kwargs`: Per-instance generation parameters
   - `response`: Model-generated response (populated after API call)

3. **EvalResult** - 3 fields documented:
   - `task`: Name of the evaluated task
   - `metrics`: Computed metric scores
   - `num_samples`: Total evaluated samples

4. **APIConfig** - 9 fields documented:
   - `base_url`: Full URL to chat completions endpoint
   - `model`: Model name for API requests
   - `api_key`: Bearer token for Authorization header
   - `max_tokens`: Default max tokens per request
   - `temperature`: Sampling temperature
   - `seed`: Random seed for reproducibility
   - `num_concurrent`: Max concurrent HTTP requests
   - `timeout`: Request timeout in seconds
   - `max_retries`: Retry attempts on failure

## Known Limitations
- No local model inference (by design)
- No multi-GPU support (by design)
- No request/response caching (removed)
- No train/val/test split distinction (by design - for inference framework comparisons)
- `datasets` library still required for loading benchmarks
- Some task-specific utils still have torch/transformers imports (in `lm_eval/tasks/`)
- No W&B or Zeno result visualization (removed)
- No HuggingFace evaluate metrics fallback (removed)
- Task-specific optional dependencies removed (ifeval, math, multilingual) - tasks using these may have reduced functionality
- Minimal implementation (`lm_eval_mini.py`) only supports GSM8K and ChartQA explicitly

### Phase 13: Simplify LocalCompletionsAPI and Improve Naming

Cleaned up `LocalCompletionsAPI` class in `lm_eval_mini.py` to remove unnecessary complexity inherited from the larger framework:

1. **Removed unused `generate` flag:**
   - The class had dual-mode functionality (generate vs logprobs) but logprobs mode was never used
   - Removed `generate` parameter from `_create_payload()`, `model_call()`, `get_batched_requests()`
   - Removed the dead logprobs code path that returned `{"echo": True, "logprobs": 1, ...}`

2. **Renamed private fields for consistency with `APIConfig`:**
   - `_seed` → `seed`
   - `_max_gen_toks` → `max_tokens` (clearer than abbreviation "toks")
   - `_concurrent` → `num_concurrent`

3. **Simplified return type of `get_batched_requests()`:**
   - Was: `list[list[str] | list[tuple[float, bool]]]` (union for logprobs mode)
   - Now: `list[list[str]]` (only generation mode)

4. **Clarified docstring:**
   - Added note explaining this is for text completions (`/v1/completions`) vs chat completions used by the main evaluation pipeline

5. **Fixed async mock bug in `test_chartqa_integration.py`:**
   - The mock for `session.post()` was returning a coroutine, not an async context manager
   - Real code uses `async with session.post(...) as resp:` which requires `__aenter__`/`__aexit__`
   - Fixed by creating proper `MockContextManager` class

6. **Updated tests:**
   - Removed `generate=True` arguments from test calls
   - All 18 mini tests passing
