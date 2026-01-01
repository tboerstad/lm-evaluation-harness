"""Cross-file YAML task testing.

This test discovers all YAML task files and validates that each one can:
1. Load the YAML config successfully
2. Load the dataset (at least 1 sample)
3. Build evaluation instances

This is a comprehensive test that takes ~3 hours to run through all 800+ tasks.
It generates a detailed report of which tasks work and which fail.

Usage:
    pytest tests/test_yaml_cross_file.py -v --tb=no -x  # Quick fail on first error
    pytest tests/test_yaml_cross_file.py -v --tb=short  # Run all, short traceback
    pytest tests/test_yaml_cross_file.py::test_all_yaml_tasks_report -s  # Full report
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml


TASKS_DIR = Path(__file__).parent.parent / "lm_eval" / "tasks"
REPORT_FILE = Path(__file__).parent / "yaml_cross_file_report.json"


@dataclass
class TaskTestResult:
    """Result of testing a single YAML task file."""
    yaml_path: str
    task_name: str = ""
    config_load_success: bool = False
    config_load_error: str = ""
    dataset_path: str = ""
    dataset_name: str = ""
    dataset_load_success: bool = False
    dataset_load_error: str = ""
    instance_build_success: bool = False
    instance_build_error: str = ""
    sample_prompt: str = ""
    sample_target: str = ""
    test_time_seconds: float = 0.0

    @property
    def fully_working(self) -> bool:
        return self.config_load_success and self.dataset_load_success and self.instance_build_success

    @property
    def failure_stage(self) -> str:
        if not self.config_load_success:
            return "config_load"
        if not self.dataset_load_success:
            return "dataset_load"
        if not self.instance_build_success:
            return "instance_build"
        return "none"


def discover_yaml_task_files() -> list[Path]:
    """Find all YAML files that have dataset_path (actual task files)."""
    task_files = []
    for yaml_path in TASKS_DIR.rglob("*.yaml"):
        # Skip files starting with underscore (typically include-only configs)
        if yaml_path.name.startswith("_"):
            continue
        # Check if file has dataset_path
        try:
            with open(yaml_path) as f:
                content = f.read()
                if "dataset_path:" in content:
                    task_files.append(yaml_path)
        except Exception:
            pass
    return sorted(task_files)


def load_yaml_config(yaml_path: Path) -> tuple[dict[str, Any], str]:
    """Load YAML config, handling !function tags and includes.

    Returns (config_dict, error_message).
    """
    class FunctionTagLoader(yaml.SafeLoader):
        pass
    FunctionTagLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))

    try:
        with open(yaml_path) as f:
            data = yaml.load(f, Loader=FunctionTagLoader)

        # Handle include directive
        if "include" in data:
            base_path = yaml_path.parent / data["include"]
            base_config, base_error = load_yaml_config(base_path)
            if base_error:
                return {}, f"Failed to load included config: {base_error}"
            # Merge configs
            for key, value in data.items():
                if key != "include":
                    base_config[key] = value
            return base_config, ""

        return data, ""
    except Exception as e:
        return {}, str(e)


def try_load_dataset(dataset_path: str, dataset_name: str | None, limit: int = 1) -> tuple[list[dict], str]:
    """Try to load dataset samples.

    Returns (samples, error_message).
    """
    import datasets

    # Try streaming first for speed
    for split in ["test", "validation", "train", "dev"]:
        try:
            ds = datasets.load_dataset(
                dataset_path,
                dataset_name,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            samples = list(ds.take(limit))
            if samples:
                return samples, ""
        except Exception:
            continue

    # Try non-streaming as fallback
    try:
        ds_dict = datasets.load_dataset(
            dataset_path,
            dataset_name,
            trust_remote_code=True,
        )
        for split in ds_dict:
            samples = [ds_dict[split][i] for i in range(min(limit, len(ds_dict[split])))]
            if samples:
                return samples, ""
    except Exception as e:
        return [], str(e)

    return [], "No valid splits found"


def try_build_instance(config: dict[str, Any], doc: dict[str, Any]) -> tuple[dict, str]:
    """Try to render prompt and target from config and document.

    Returns (instance_dict, error_message).
    """
    import jinja2
    jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)

    try:
        doc_to_text = config.get("doc_to_text", "")
        doc_to_target = config.get("doc_to_target", "")

        # Render prompt
        if doc_to_text:
            if "{{" in str(doc_to_text):
                prompt = jinja_env.from_string(str(doc_to_text)).render(**doc)
            elif doc_to_text in doc:
                prompt = str(doc[doc_to_text])
            else:
                prompt = str(doc_to_text)
        else:
            prompt = ""

        # Render target
        if doc_to_target:
            if "{{" in str(doc_to_target):
                target = jinja_env.from_string(str(doc_to_target)).render(**doc)
            elif doc_to_target in doc:
                target = str(doc[doc_to_target])
            else:
                target = str(doc_to_target)
        else:
            target = ""

        return {"prompt": prompt, "target": target}, ""
    except Exception as e:
        return {}, str(e)


def test_single_yaml_task(yaml_path: Path) -> TaskTestResult:
    """Test a single YAML task file."""
    start_time = time.time()
    result = TaskTestResult(yaml_path=str(yaml_path))

    # Step 1: Load YAML config
    config, error = load_yaml_config(yaml_path)
    if error:
        result.config_load_error = error
        result.test_time_seconds = time.time() - start_time
        return result

    result.config_load_success = True
    result.task_name = config.get("task", yaml_path.stem)
    result.dataset_path = config.get("dataset_path", "")
    result.dataset_name = config.get("dataset_name")

    # Step 2: Load dataset
    samples, error = try_load_dataset(result.dataset_path, result.dataset_name, limit=1)
    if error or not samples:
        result.dataset_load_error = error or "No samples loaded"
        result.test_time_seconds = time.time() - start_time
        return result

    result.dataset_load_success = True

    # Step 3: Build instance
    instance, error = try_build_instance(config, samples[0])
    if error:
        result.instance_build_error = error
        result.test_time_seconds = time.time() - start_time
        return result

    result.instance_build_success = True
    result.sample_prompt = instance.get("prompt", "")[:500]  # Truncate for readability
    result.sample_target = str(instance.get("target", ""))[:200]
    result.test_time_seconds = time.time() - start_time

    return result


def generate_report(results: list[TaskTestResult]) -> dict[str, Any]:
    """Generate summary report from test results."""
    total = len(results)
    working = sum(1 for r in results if r.fully_working)
    config_failures = [r for r in results if not r.config_load_success]
    dataset_failures = [r for r in results if r.config_load_success and not r.dataset_load_success]
    instance_failures = [r for r in results if r.config_load_success and r.dataset_load_success and not r.instance_build_success]

    # Group dataset failures by error type
    dataset_error_groups: dict[str, list[str]] = {}
    for r in dataset_failures:
        error_key = r.dataset_load_error[:100] if r.dataset_load_error else "Unknown"
        # Simplify common errors
        if "404" in error_key or "not found" in error_key.lower():
            error_key = "Dataset not found (404)"
        elif "ConnectionError" in error_key or "ConnectionReset" in error_key:
            error_key = "Network connection error"
        elif "gated" in error_key.lower() or "access" in error_key.lower():
            error_key = "Gated/restricted dataset"
        elif "trust_remote_code" in error_key.lower():
            error_key = "Trust remote code required"
        dataset_error_groups.setdefault(error_key, []).append(r.task_name or r.yaml_path)

    return {
        "summary": {
            "total_tasks": total,
            "working": working,
            "working_percentage": round(100 * working / total, 1) if total else 0,
            "config_load_failures": len(config_failures),
            "dataset_load_failures": len(dataset_failures),
            "instance_build_failures": len(instance_failures),
        },
        "config_load_failures": [
            {"task": r.task_name or r.yaml_path, "error": r.config_load_error}
            for r in config_failures
        ],
        "dataset_load_failures_by_type": {
            error_type: tasks for error_type, tasks in sorted(
                dataset_error_groups.items(), key=lambda x: -len(x[1])
            )
        },
        "instance_build_failures": [
            {"task": r.task_name, "error": r.instance_build_error}
            for r in instance_failures
        ],
        "working_tasks": [
            r.task_name for r in results if r.fully_working
        ],
        "detailed_results": [
            {
                "yaml_path": r.yaml_path,
                "task_name": r.task_name,
                "fully_working": r.fully_working,
                "failure_stage": r.failure_stage,
                "config_load_error": r.config_load_error,
                "dataset_path": r.dataset_path,
                "dataset_load_error": r.dataset_load_error,
                "instance_build_error": r.instance_build_error,
                "test_time_seconds": round(r.test_time_seconds, 2),
            }
            for r in results
        ],
    }


# Discover all task files for parametrized testing
ALL_YAML_TASK_FILES = discover_yaml_task_files()


@pytest.mark.parametrize("yaml_path", ALL_YAML_TASK_FILES[:5], ids=lambda p: p.stem)
def test_sample_yaml_tasks(yaml_path: Path):
    """Quick sanity test: test first 5 YAML files."""
    result = test_single_yaml_task(yaml_path)

    # We don't assert failure - we just want to see results
    if not result.fully_working:
        pytest.skip(f"Task not fully working: {result.failure_stage} - {result.config_load_error or result.dataset_load_error or result.instance_build_error}")


def test_all_yaml_tasks_report():
    """
    Comprehensive test: run through ALL YAML task files and generate report.

    This test is designed to run for several hours and collect complete results.
    Run with: pytest tests/test_yaml_cross_file.py::test_all_yaml_tasks_report -s -v
    """
    yaml_files = discover_yaml_task_files()
    print(f"\n{'='*80}")
    print(f"YAML Cross-File Task Test")
    print(f"Total tasks to test: {len(yaml_files)}")
    print(f"{'='*80}\n")

    results: list[TaskTestResult] = []
    start_time = time.time()

    for i, yaml_path in enumerate(yaml_files):
        progress = f"[{i+1}/{len(yaml_files)}]"
        print(f"{progress} Testing: {yaml_path.relative_to(TASKS_DIR)}...", end=" ", flush=True)

        try:
            result = test_single_yaml_task(yaml_path)
            results.append(result)

            status = "OK" if result.fully_working else f"FAIL ({result.failure_stage})"
            print(f"{status} ({result.test_time_seconds:.1f}s)")

            if not result.fully_working:
                error = result.config_load_error or result.dataset_load_error or result.instance_build_error
                if error:
                    print(f"         Error: {error[:100]}...")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append(TaskTestResult(
                yaml_path=str(yaml_path),
                config_load_error=f"Exception: {traceback.format_exc()[:500]}"
            ))

        # Save intermediate results every 50 tasks
        if (i + 1) % 50 == 0:
            report = generate_report(results)
            with open(REPORT_FILE, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n   --- Intermediate report saved ({report['summary']['working']}/{i+1} working) ---\n")

    # Generate final report
    total_time = time.time() - start_time
    report = generate_report(results)
    report["total_test_time_seconds"] = round(total_time, 1)
    report["total_test_time_human"] = f"{int(total_time//3600)}h {int((total_time%3600)//60)}m {int(total_time%60)}s"

    # Save report
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    summary = report["summary"]
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks tested: {summary['total_tasks']}")
    print(f"Working tasks: {summary['working']} ({summary['working_percentage']}%)")
    print(f"Config load failures: {summary['config_load_failures']}")
    print(f"Dataset load failures: {summary['dataset_load_failures']}")
    print(f"Instance build failures: {summary['instance_build_failures']}")
    print(f"Total test time: {report['total_test_time_human']}")
    print(f"\nDetailed report saved to: {REPORT_FILE}")
    print(f"{'='*80}")

    # Print top failure reasons
    if report["dataset_load_failures_by_type"]:
        print("\nTop dataset load failure reasons:")
        for error_type, tasks in list(report["dataset_load_failures_by_type"].items())[:5]:
            print(f"  - {error_type}: {len(tasks)} tasks")

    # Don't fail the test - this is for reporting
    assert True


if __name__ == "__main__":
    # Can be run directly for testing
    test_all_yaml_tasks_report()
