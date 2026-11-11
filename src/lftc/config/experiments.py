"""Plain experiment recipes and Cartesian comparison grids."""
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml

from lftc.domain import ConfigError

MODEL_IDS = ("tfidf_lr", "tfidf_svm", "w2v_svm", "bilstm", "distilbert", "modernbert")


def load_experiment_config(path: Path) -> dict:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read experiment {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ConfigError("Experiment configuration requires schema_version: 1")
    expand_experiment_config(data)
    return data


def expand_experiment_config(config: dict) -> list[dict]:
    """Expand a single recipe or a dataset/model/seed/sampling comparison grid."""
    datasets = config.get("datasets", [config.get("dataset")])
    models = config.get("models", [config.get("model")])
    seeds = config.get("seeds", [config.get("seed", 42)])
    conditions = config.get("conditions", [config.get("sampling", {"mode": "all"})])
    for name, values in (("datasets", datasets), ("models", models),
                         ("seeds", seeds), ("conditions", conditions)):
        if not isinstance(values, list) or not values:
            raise ConfigError(f"{name} must be a nonempty list")
    if any(not isinstance(value, str) or not value for value in datasets):
        raise ConfigError("Every experiment requires a dataset id")
    if any(value not in MODEL_IDS for value in models):
        raise ConfigError(f"model must be one of {', '.join(MODEL_IDS)}")
    if any(type(seed) is not int for seed in seeds):
        raise ConfigError("seeds must contain integers")
    if any(not isinstance(condition, dict) for condition in conditions):
        raise ConfigError("conditions must contain sampling mappings")
    for name in ("data", "training", "evaluation"):
        if not isinstance(config.get(name, {}), dict):
            raise ConfigError(f"{name} must be a mapping")
    result = []
    for dataset, model, seed, condition in product(datasets, models, seeds, conditions):
        row = deepcopy(config)
        for key in ("datasets", "models", "seeds", "conditions"):
            row.pop(key, None)
        row.update(dataset=dataset, model=model, seed=seed, sampling=deepcopy(condition))
        result.append(row)
    return result
