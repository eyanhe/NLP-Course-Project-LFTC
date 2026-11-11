"""Create a small source-only ZIP without walking local data or environments."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

SOURCE_DIRECTORIES = ("src", "configs", "apps", "ops", "docs", "tests")
SOURCE_FILES = ("README.md", "pyproject.toml", ".gitignore")
EXCLUDED = {
    "artifacts", "node_modules", "__pycache__", ".cache", "cache",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tools", ".git", ".superpowers",
}


def _source_files(directory: Path):
    for child in sorted(directory.iterdir()):
        # Skip links rather than following a directory outside the source tree.
        if child.is_symlink() or child.name in EXCLUDED or child.name.startswith(".venv"):
            continue
        if child.is_dir():
            yield from _source_files(child)
        elif child.is_file() and child.suffix not in {".pyc", ".pyo"}:
            yield child


def package_source(root: Path, destination: Path) -> Path:
    """Package maintained directories and project metadata, never runtime assets."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            path = root / name
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(root).as_posix())
        for name in SOURCE_DIRECTORIES:
            directory = root / name
            if directory.is_dir():
                for path in _source_files(directory):
                    if path.resolve() != destination:
                        archive.write(path, path.relative_to(root).as_posix())
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="Destination ZIP file")
    args = parser.parse_args(argv)
    print(package_source(args.root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
