"""Shared types; no dependency on configuration, models or application code."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class LftcError(Exception):
    """Base error exposed by the command-line interface."""


class ConfigError(LftcError):
    """A configuration file cannot be read or interpreted."""


class PathBoundaryError(LftcError):
    """A relative path leaves its configured directory."""


class TaskType(str, Enum):
    SINGLE_LABEL = "single_label"
    MULTILABEL = "multilabel"
    PAIR_CLASSIFICATION = "pair_classification"


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    config_dir: Path
    artifacts_dir: Path
    assets_dir: Path
    prepared_dir: Path
    runs_dir: Path
    results_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class DatasetAsset:
    id: str
    directory: str
    repo: str
    revision: str
    files: tuple[str, ...]
    task: TaskType
    label_count: int
    label_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelAsset:
    id: str
    directory: str
    repo: str
    revision: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class AssetsCatalog:
    datasets: tuple[DatasetAsset, ...]
    models: tuple[ModelAsset, ...]


@dataclass(frozen=True)
class Example:
    sample_id: str
    text: str
    labels: tuple[int, ...]
    split: str
    text_pair: str | None = None
    dataset_id: str = ""


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    task: TaskType
    label_count: int
    label_names: tuple[str, ...]
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]
