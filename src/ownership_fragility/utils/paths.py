from __future__ import annotations

from pathlib import Path
import os


def project_root() -> Path:
    """Return the project root from ONF_PROJECT_ROOT or the current working directory."""
    return Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
