"""Test that validates task configs work with liteval - one per unique dataset."""

import logging
from pathlib import Path

from liteval import TaskConfig, build_instances

# Reduce logging noise
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)


def find_unique_tasks(tasks_dir: Path) -> dict[str, Path]:
    """Find one YAML per unique (dataset_path, dataset_name) combo."""
    seen: dict[str, Path] = {}

    for yaml_path in sorted(tasks_dir.rglob("*.yaml")):
        # Skip template files
        if yaml_path.name.startswith("_"):
            continue
        if yaml_path.name in ("group.yaml", "groups.yaml"):
            continue

        try:
            config = TaskConfig.from_yaml(yaml_path)
            if not config.dataset_path:
                continue

            key = f"{config.dataset_path}|{config.dataset_name or ''}"
            if key not in seen:
                seen[key] = yaml_path
        except Exception:
            # Track failed config loads separately
            seen[f"__error__{yaml_path}"] = yaml_path

    return seen


def test_single_task(yaml_path: Path) -> tuple[str, str]:
    """Test loading a single task config and building one instance.

    Returns (status, message) where status is 'ok', 'skip', or 'fail'.
    """
    import datasets

    try:
        config = TaskConfig.from_yaml(yaml_path)

        if not config.dataset_path:
            return "skip", "no dataset_path"

        # Try to load one sample
        for split in ["test", "validation", "train"]:
            try:
                ds = datasets.load_dataset(
                    config.dataset_path,
                    config.dataset_name,
                    split=split,
                    streaming=True,
                )
                docs = list(ds.take(1))
                if docs:
                    break
            except (ValueError, KeyError):
                continue
        else:
            return "fail", "no valid split"

        # Build instance
        instances = build_instances(config, docs, limit=1)
        if not instances:
            return "fail", "no instances built"

        if not instances[0].prompt:
            return "fail", "empty prompt"

        return "ok", f"dataset={config.dataset_path}"

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)[:80].replace("\n", " ")
        return "fail", f"{error_type}: {error_msg}"


def main():
    """Run validation on unique tasks and print summary."""
    tasks_dir = Path(__file__).parent.parent / "tasks"

    print("Scanning task configs...")
    unique_tasks = find_unique_tasks(tasks_dir)
    print(f"Found {len(unique_tasks)} unique dataset configurations\n")

    results = {"ok": [], "skip": [], "fail": []}

    for i, (key, yaml_path) in enumerate(unique_tasks.items()):
        rel_path = yaml_path.relative_to(tasks_dir)

        if key.startswith("__error__"):
            status, msg = "fail", "config load error"
        else:
            status, msg = test_single_task(yaml_path)

        results[status].append((rel_path, msg))

        # Progress
        icon = {"ok": "✓", "skip": "○", "fail": "✗"}[status]
        print(f"[{i+1}/{len(unique_tasks)}] {icon} {rel_path}")
        if status == "fail":
            print(f"    → {msg}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Unique datasets tested: {len(unique_tasks)}")
    print(f"  ✓ OK:      {len(results['ok'])}")
    print(f"  ○ Skipped: {len(results['skip'])}")
    print(f"  ✗ Failed:  {len(results['fail'])}")

    if results["fail"]:
        print(f"\n{'=' * 60}")
        print("FAILURE BREAKDOWN")
        print("=" * 60)

        # Group by error type
        by_error: dict[str, list] = {}
        for path, msg in results["fail"]:
            error_type = msg.split(":")[0] if ":" in msg else msg
            by_error.setdefault(error_type, []).append((path, msg))

        for error_type, tasks in sorted(by_error.items(), key=lambda x: -len(x[1])):
            print(f"\n{error_type} ({len(tasks)}):")
            for path, _ in tasks[:3]:
                print(f"  - {path}")
            if len(tasks) > 3:
                print(f"  ... and {len(tasks) - 3} more")

    # Return failure count
    return len(results["fail"])


if __name__ == "__main__":
    exit(main())
