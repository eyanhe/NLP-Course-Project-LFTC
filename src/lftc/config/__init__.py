"""Load workspace paths and the dataset/model catalog."""
import os
from pathlib import Path, PureWindowsPath
import tomllib

import yaml

from lftc.domain import (
    AssetsCatalog, ConfigError, DatasetAsset, ModelAsset, PathBoundaryError,
    TaskType, WorkspacePaths,
)
from lftc.governance import bounded_path
from lftc.config.experiments import expand_experiment_config, load_experiment_config


def _project_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        with (path / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read project at {path}: {exc}") from exc
    if project.get("project", {}).get("name") != "lftc":
        raise ConfigError(f"Not an lftc project: {path}")
    return path


def resolve_project_root(start: Path | None = None) -> Path:
    """Use LFTC_ROOT or find the nearest lftc pyproject above start/cwd."""
    if override := os.environ.get("LFTC_ROOT"):
        return _project_root(Path(override))
    current = (start if start is not None else Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            try:
                return _project_root(candidate)
            except ConfigError:
                continue
    raise ConfigError("Cannot find lftc project; use --root or LFTC_ROOT")


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        raise ConfigError(f"Expected a mapping with schema_version: 1 in {path}")
    if data["schema_version"] != 1:
        raise ConfigError(f"Unsupported schema_version in {path}")
    return data


def _relative_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError("Expected a nonempty relative path")
    if Path(value).is_absolute() or PureWindowsPath(value).drive:
        raise PathBoundaryError(f"Expected a relative path: {value}")
    return bounded_path(value, root)


def _environment_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_paths_config(root: Path | None = None) -> WorkspacePaths:
    root = resolve_project_root() if root is None else _project_root(root)
    config_dir = _environment_path(os.environ.get("LFTC_CONFIG_DIR", "configs"), root)
    data = _read_yaml(config_dir / "paths.yaml")
    if override := os.environ.get("LFTC_ARTIFACTS_DIR"):
        artifacts_dir = _environment_path(override, root)
    else:
        artifacts_dir = _relative_path(data.get("artifacts_dir", "artifacts"), root)
    children = {
        f"{name}_dir": _relative_path(data.get(f"{name}_dir", name), artifacts_dir)
        for name in ("assets", "prepared", "runs", "results", "reports")
    }
    return WorkspacePaths(root, config_dir, artifacts_dir, **children)


def load_assets_config(config_path: Path) -> AssetsCatalog:
    data = _read_yaml(config_path)
    groups = {}
    try:
        for group, asset_type in (("datasets", DatasetAsset), ("models", ModelAsset)):
            rows = data[group]
            if not isinstance(rows, list):
                raise ValueError(f"{group} must be a list")
            assets = []
            ids = set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"{group} entries must be mappings")
                fields = {key: row[key] for key in ("id", "directory", "repo", "revision")}
                if any(not isinstance(value, str) or not value for value in fields.values()):
                    raise ValueError("Asset metadata must contain nonempty strings")
                if fields["id"] in ids:
                    raise ValueError(f"Duplicate asset id: {fields['id']}")
                ids.add(fields["id"])
                _relative_path(fields["directory"], Path.cwd())
                files = row["files"]
                if not isinstance(files, list) or not files:
                    raise ValueError("files must be a nonempty list")
                for name in files:
                    _relative_path(name, Path.cwd())
                fields["files"] = tuple(files)
                if group == "datasets":
                    count = row["label_count"]
                    if type(count) is not int or count < 2:
                        raise ValueError("label_count must be an integer >= 2")
                    labels = row.get("label_names", [])
                    if not isinstance(labels, list) or (labels and len(labels) != count):
                        raise ValueError("label_names must match label_count")
                    fields.update(task=TaskType(row["task"]), label_count=count,
                                  label_names=tuple(labels))
                assets.append(asset_type(**fields))
            groups[group] = tuple(assets)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid asset catalog {config_path}: {exc}") from exc
    return AssetsCatalog(**groups)
