"""Small immutable records shared by preparation, manifests and later learners."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Protocol = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Example:
    dataset_id: str
    sample_id: str
    text: str
    labels: tuple[int, ...]
    official_split: str
    group_id: str = ''
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    dataset_id: str
    task: str
    label_names: tuple[str, ...]
    partitions: dict[str, tuple[Example, ...]]
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class VisibleInput:
    sample_id: str
    text: str
    start_char: int
    end_char: int
    content_tokens: int
    full_tokens: int
    text_hash: str


@dataclass(frozen=True, slots=True)
class Condition:
    condition_id: str
    experiment_id: str
    dataset_id: str
    train_ids: tuple[str, ...]
    data_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    condition_id: str
    model_id: str
    training_seed: int | None
    hparams: dict[str, Any]
    input_hash: str
    asset_hash: str
    code_hash: str
    strict_no_val: bool


@dataclass(frozen=True, slots=True)
class PredictionRow:
    run_id: str
    sample_id: str
    group_id: str
    labels: tuple[int, ...]
    prediction: tuple[int, ...]
    scores: tuple[float, ...]
    score_kind: str
    input_hash: str
