from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from a file or directory inside the repo."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError(f"Could not find repository root from {current}")


REPO_ROOT = find_repo_root()
CODE_ROOT = REPO_ROOT / "code"
DATA_ROOT = REPO_ROOT / "data"
EXTERNAL_CODE_ROOT = CODE_ROOT / "external"
