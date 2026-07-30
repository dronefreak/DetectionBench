#!/usr/bin/env python3
"""Generate PR body from git history."""

import os
import subprocess
import sys


def run_cmd(cmd: list[str]) -> str:
    """
    Run command and return stdout.

    Note: Commands use validated inputs only. The BASE_REF is validated
    in main() before being used in any git commands.
    """
    result = subprocess.run(  # nosec B603  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=True,
        shell=False,  # Explicit: never use shell
    )
    return result.stdout.strip()


def categorize_files(files: list[str]) -> dict[str, int]:
    """Categorize changed files."""
    categories = {
        "src": 0,
        "test": 0,
        "doc": 0,
        "ci": 0,
    }

    for f in files:
        if f.startswith("src/"):
            categories["src"] += 1
        elif f.startswith("tests/"):
            categories["test"] += 1
        elif f.endswith("README.md") or (".github/" in f and f.endswith(".md")):
            categories["doc"] += 1
        elif ".github/workflows/" in f or ".github/dependabot" in f:
            categories["ci"] += 1

    return categories


def build_summary_line(categories: dict[str, int], total: int) -> str:
    """Build human-readable summary of changes."""
    parts = []
    if categories["src"] > 0:
        parts.append(f"{categories['src']} source file(s)")
    if categories["test"] > 0:
        parts.append(f"{categories['test']} test file(s)")
    if categories["doc"] > 0:
        parts.append(f"{categories['doc']} doc file(s)")
    if categories["ci"] > 0:
        parts.append(f"{categories['ci']} CI file(s)")

    if parts:
        return f"This PR touches {', '.join(parts)} across {total} file(s) total."
    return f"This PR modifies {total} file(s)."


def main() -> int:
    """Generate PR body from git history."""
    base_ref = os.environ["BASE_REF"]

    # Validate BASE_REF to prevent command injection
    # Valid git refs: alphanumeric, hyphens, underscores, slashes, dots
    # Reject anything suspicious
    import re

    if not re.match(r"^[a-zA-Z0-9/_.-]+$", base_ref):
        print(f"ERROR: Invalid BASE_REF format: {base_ref}", file=sys.stderr)
        return 1

    # Additional safety: reject refs that could be command injection attempts
    if base_ref.startswith("-") or ".." in base_ref:
        print(f"ERROR: Suspicious BASE_REF detected: {base_ref}", file=sys.stderr)
        return 1

    # Get merge base
    merge_base = run_cmd(["git", "merge-base", f"origin/{base_ref}", "HEAD"])

    # Get commit messages
    commits = run_cmd(["git", "log", "--pretty=format:- %s", f"{merge_base}..HEAD"])

    # Get changed files
    diff_stat = run_cmd(["git", "diff", "--stat", f"{merge_base}..HEAD"])
    files = run_cmd(["git", "diff", "--name-only", f"{merge_base}..HEAD"]).split("\n")
    files = [f for f in files if f]  # Remove empty strings

    # Categorize
    categories = categorize_files(files)
    summary = build_summary_line(categories, len(files))

    # Build body
    body = f"""## Summary

{summary}

## Changes

{commits}

<details>
<summary>Diff stats</summary>
```
{diff_stat}
```

</details>

## Testing

- [ ] Tests pass locally (`python -m pytest tests/ -v`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)

## Related Issues

<!-- Link any related issues: Fixes #123, Closes #456 -->
"""

    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
