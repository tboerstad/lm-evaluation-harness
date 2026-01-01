# Agent Instructions

## Setup (Run First)

The `.venv` directory is gitignored and won't exist in fresh sessions. Create it before making any changes:

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
