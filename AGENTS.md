# Agent Instructions

## Setup (MUST RUN FIRST)

`.venv` is gitignored and DOES NOT EXIST in fresh sessions. Create it before making any changes:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pre-commit install
```

## Workflow

Pre-commit hooks run automatically on `git commit`:

- **ruff**: Linting and formatting (auto-fixes where possible)
- **pytest**: Runs test suite

If hooks fail:
1. If files were modified by ruff, stage them (`git add -u`) and commit again
2. If tests fail, fix the code and retry

## Manual Checks

Run checks without committing:

```bash
ruff check . --fix
ruff format .
pytest
```

## Code Style

- **No local imports**: Use module-level imports only. No imports inside functions or methods.
- **High SNR comments only**: Comments should be useful for future readers who lack context. Don't add comments that only make sense in the context of our conversation (e.g., "Required - set via CLI" or "Fixed per review"). If a comment wouldn't help someone reading the code cold, don't add it.
