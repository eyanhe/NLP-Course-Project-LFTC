"""Common workspace path handling, independent of configuration and experiments."""
from pathlib import Path

from lftc.domain import PathBoundaryError


def bounded_path(candidate: Path | str, root: Path) -> Path:
    """Resolve a path within root, including existing symbolic links."""
    root = root.resolve()
    try:
        value = Path(candidate)
        resolved = (value if value.is_absolute() else root / value).resolve()
    except (OSError, ValueError) as exc:
        raise PathBoundaryError(f"Invalid path: {candidate}") from exc
    if not resolved.is_relative_to(root):
        raise PathBoundaryError(f"Path leaves {root}: {candidate}")
    return resolved
