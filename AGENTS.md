# Agent Instructions

## Setup (Run First)

Before making any changes, install dev dependencies and pre-commit hooks:

```bash
pip install -e ".[dev]"
pre-commit install
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
