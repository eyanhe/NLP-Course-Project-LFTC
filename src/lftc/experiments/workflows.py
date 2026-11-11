"""Experiment workflows composed from the lower package layers."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from lftc.config import (
    expand_experiment_config, load_assets_config, load_experiment_config, load_paths_config,
)
from lftc.domain import ConfigError, Example, TaskType
from lftc.governance import bounded_path


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _context(root=None):
    paths = load_paths_config(Path(root) if root is not None else None)
    catalog = load_assets_config(paths.config_dir / "assets.yaml")
    return paths, catalog


def _asset(catalog, dataset_id):
    for asset in catalog.datasets:
        if asset.id == dataset_id:
            return asset
    raise ConfigError(f"Unknown dataset: {dataset_id}")


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise ConfigError("run_id must use letters, digits, underscores, dots or hyphens")
    return value


def _run_dir(paths, run_id):
    directory = bounded_path(_safe_id(run_id), paths.runs_dir)
    if not (directory / "run.json").is_file():
        raise ConfigError(f"Saved run not found: {run_id}")
    return directory


def _labels(examples, task, label_count):
    import numpy as np
    if TaskType(task) == TaskType.MULTILABEL:
        values = np.zeros((len(examples), label_count), dtype=int)
        for index, row in enumerate(examples):
            values[index, list(row.labels)] = 1
        return values
    return np.asarray([row.labels[0] for row in examples], dtype=int)


def matrix(config_path: Path, root=None, output: Path | None = None) -> list[dict]:
    paths, catalog = _context(root)
    configs = expand_experiment_config(load_experiment_config(config_path))
    for config in configs:
        _asset(catalog, config["dataset"])
    if output:
        _write_json(Path(output), configs)
    return configs


def prepare(config_path: Path, root=None) -> list[dict]:
    from lftc.data import save_prepared
    paths, catalog = _context(root)
    summaries = []
    seen = set()
    for config in expand_experiment_config(load_experiment_config(config_path)):
        key = (config["dataset"], config["seed"], json.dumps(config.get("data", {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        dataset = _load_data(config, paths, catalog)
        fraction = float(config.get("data", {}).get("validation_fraction", 0.15))
        destination = paths.prepared_dir / f"{dataset.dataset_id}-seed{config['seed']}-val{fraction}"
        if config.get("data", {}).get("limit_per_split") is not None:
            destination = destination.with_name(destination.name + f"-limit{config['data']['limit_per_split']}")
        save_prepared(dataset, destination)
        summaries.append({"dataset": dataset.dataset_id, "directory": str(destination),
                          "train": len(dataset.train), "validation": len(dataset.validation),
                          "test": len(dataset.test)})
    return summaries


def _load_data(config, paths, catalog):
    from lftc.data import load_dataset
    data = config.get("data", {})
    return load_dataset(_asset(catalog, config["dataset"]), paths.assets_dir,
                        limit_per_split=data.get("limit_per_split"),
                        validation_fraction=data.get("validation_fraction", 0.15),
                        seed=config["seed"])


def run(config_path: Path, root=None, run_id: str | None = None):
    """Train the explicitly selected recipes; never alter historical artifacts."""
    paths, catalog = _context(root)
    configs = expand_experiment_config(load_experiment_config(config_path))
    if run_id and len(configs) != 1:
        raise ConfigError("--run-id requires a single experiment recipe")
    # Resolve all dataset ids before starting a grid.
    for config in configs:
        _asset(catalog, config["dataset"])
    results = [_run_one(config, paths, catalog, run_id) for config in configs]
    return results[0] if len(results) == 1 else results


def _run_one(config, paths, catalog, run_id=None):
    from lftc.data import sample_training
    from lftc.evaluation import metric_bundle, select_threshold
    from lftc.training import save_model, train_model
    import numpy as np

    created = datetime.now(timezone.utc)
    identifier = run_id or "-".join((
        re.sub(r"[^A-Za-z0-9_.-]+", "-", config.get("name", "experiment")),
        config["dataset"], config["model"], str(config["seed"]),
        created.strftime("%Y%m%dT%H%M%S%f"),
    ))
    directory = bounded_path(_safe_id(identifier), paths.runs_dir)
    if directory.exists():
        raise ConfigError(f"Output already exists; choose another run id: {identifier}")
    start = time.perf_counter()
    dataset = _load_data(config, paths, catalog)
    sampling = dict(config.get("sampling", {}))
    sampling.pop("seed", None)
    train = sample_training(dataset.train, seed=config["seed"], **sampling)
    if not train or not dataset.test:
        raise ConfigError("Training and test partitions must contain labeled rows")
    source = next((asset for asset in catalog.models if asset.id == config["model"]), None)
    model_dir = bounded_path(source.directory, paths.assets_dir) if source else None
    options = {**config.get("training", {}), "seed": config["seed"]}
    prepared_at = time.perf_counter()
    predictor = train_model(train, dataset.validation, dataset.task, dataset.label_count,
                            config["model"], options=options, model_dir=model_dir)
    trained = time.perf_counter()
    evaluation = config.get("evaluation", {})
    threshold = evaluation.get("threshold")
    if dataset.task == TaskType.MULTILABEL and threshold is None and dataset.validation:
        threshold = select_threshold(
            _labels(dataset.validation, dataset.task, dataset.label_count),
            predictor.predict_scores(dataset.validation), predictor.score_kind,
        )
    scoring_at = time.perf_counter()
    scores = np.asarray(predictor.predict_scores(dataset.test))
    scored = time.perf_counter()
    labels = _labels(dataset.test, dataset.task, dataset.label_count)
    metrics = metric_bundle(labels, scores, dataset.task, predictor.score_kind, threshold=threshold)
    metadata = {
        "schema_version": 1, "run_id": identifier, "dataset": dataset.dataset_id,
        "task": dataset.task.value, "label_count": dataset.label_count,
        "label_names": list(dataset.label_names), "model": config["model"],
        "seed": config["seed"], "training_rows": len(train),
        "validation_rows": len(dataset.validation), "test_rows": len(dataset.test),
        "config": config, "metrics": metrics, "threshold": threshold,
        "score_kind": predictor.score_kind,
        "timing": {"preparation_seconds": prepared_at - start,
                   "training_seconds": trained - prepared_at,
                   "validation_selection_seconds": scoring_at - trained,
                   "inference_seconds": scored - scoring_at,
                   "milliseconds_per_row": 1000 * (scored - scoring_at) / len(dataset.test)},
        "created_at": created.isoformat(),
    }
    directory.mkdir(parents=True)
    save_model(predictor, directory / "model")
    if dataset.task == TaskType.MULTILABEL:
        cutoff = threshold if threshold is not None else (0.5 if predictor.score_kind == "probability" else 0.0)
        predictions = [np.flatnonzero(row >= cutoff).tolist() for row in scores]
    else:
        predictions = [[int(np.argmax(row))] for row in scores]
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for example, score, prediction in zip(dataset.test, scores, predictions):
            row = {**asdict(example), "prediction": prediction, "scores": score.tolist(),
                   "score_kind": predictor.score_kind}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    _write_json(directory / "metrics.json", metrics)
    _write_json(directory / "run.json", metadata)
    return metadata


def predict(run_id: str, texts: list[str], text_pairs=None, root=None) -> list[dict]:
    from lftc.training import load_model
    import numpy as np
    paths = load_paths_config(Path(root) if root is not None else None)
    directory = _run_dir(paths, run_id)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ConfigError("Provide at least one nonempty text")
    pairs = [None] * len(texts) if text_pairs is None else list(text_pairs)
    if len(pairs) != len(texts):
        raise ConfigError("Text and text-pair counts must match")
    if metadata["task"] == "pair_classification" and any(not pair for pair in pairs):
        raise ConfigError("This model requires both premise and hypothesis")
    examples = tuple(Example(f"input/{i}", text, (), "inference", pair)
                     for i, (text, pair) in enumerate(zip(texts, pairs)))
    predictor = load_model(directory / "model")
    scores = np.asarray(predictor.predict_scores(examples))
    threshold = metadata.get("threshold")
    if threshold is None:
        threshold = 0.5 if predictor.score_kind == "probability" else 0.0
    result = []
    names = metadata["label_names"]
    for example, score in zip(examples, scores):
        prediction = (np.flatnonzero(score >= threshold).tolist()
                      if metadata["task"] == "multilabel" else [int(score.argmax())])
        result.append({"text": example.text, "text_pair": example.text_pair,
                       "prediction": prediction, "labels": [names[i] for i in prediction],
                       "scores": score.tolist(), "score_kind": predictor.score_kind})
    return result


def _saved_predictions(directory):
    with (directory / "predictions.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_run(run_id: str, root=None) -> dict:
    from lftc.evaluation import metric_bundle
    paths = load_paths_config(Path(root) if root is not None else None)
    directory = _run_dir(paths, run_id)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    rows = _saved_predictions(directory)
    examples = tuple(Example(row["sample_id"], row.get("text", ""), tuple(row["labels"]),
                             row.get("split", "test"), row.get("text_pair")) for row in rows)
    metrics = metric_bundle(_labels(examples, metadata["task"], metadata["label_count"]),
                            [row["scores"] for row in rows], metadata["task"],
                            metadata["score_kind"], threshold=metadata.get("threshold"))
    _write_json(directory / "metrics.json", metrics)
    metadata["metrics"] = metrics
    _write_json(directory / "run.json", metadata)
    return metrics


def diagnose_run(run_id: str, root=None) -> dict:
    from lftc.evaluation import diagnose_predictions
    paths = load_paths_config(Path(root) if root is not None else None)
    directory = _run_dir(paths, run_id)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    rows = _saved_predictions(directory)
    examples = tuple(Example(row["sample_id"], row.get("text", ""), tuple(row["labels"]),
                             row.get("split", "test"), row.get("text_pair")) for row in rows)
    result = diagnose_predictions(examples, [row["scores"] for row in rows], metadata["task"],
                                  metadata["label_count"], metadata["score_kind"],
                                  threshold=metadata.get("threshold"))
    _write_json(paths.results_dir / run_id / "diagnostics.json", result)
    return result


def compare_runs(run_a: str, run_b: str, root=None, n_resamples=1000) -> dict:
    from lftc.evaluation import bootstrap_difference, mcnemar_test
    paths = load_paths_config(Path(root) if root is not None else None)
    directories = [_run_dir(paths, run_id) for run_id in (run_a, run_b)]
    metas = [json.loads((directory / "run.json").read_text(encoding="utf-8"))
             for directory in directories]
    if any(metas[0][key] != metas[1][key] for key in ("dataset", "task", "label_count", "label_names")):
        raise ConfigError("Comparison requires the same dataset, task and label order")
    records = [{row["sample_id"]: row for row in _saved_predictions(directory)}
               for directory in directories]
    if records[0].keys() != records[1].keys():
        raise ConfigError("Paired comparison requires identical test sample ids")
    keys = sorted(records[0])
    if any(records[0][key]["labels"] != records[1][key]["labels"] for key in keys):
        raise ConfigError("Paired comparison labels disagree")
    examples = tuple(Example(key, "", tuple(records[0][key]["labels"]), "test") for key in keys)
    labels = _labels(examples, metas[0]["task"], metas[0]["label_count"])
    import numpy as np
    scores = [np.asarray([record[key]["scores"] for key in keys]) for record in records]
    if metas[0]["task"] == "multilabel":
        for index, meta in enumerate(metas):
            cutoff = meta.get("threshold")
            if cutoff is None:
                cutoff = 0.5 if meta["score_kind"] == "probability" else 0.0
            scores[index] = scores[index] - cutoff
    result = bootstrap_difference(labels, scores[0], scores[1], metas[0]["task"],
                                  "logit", threshold=0.0, n_resamples=n_resamples)
    if metas[0]["task"] != "multilabel":
        result["mcnemar"] = mcnemar_test(labels,
            [records[0][key]["prediction"][0] for key in keys],
            [records[1][key]["prediction"][0] for key in keys])
    return result


def report_run(run_id: str, root=None) -> dict:
    from lftc.reporting import build_report
    paths = load_paths_config(Path(root) if root is not None else None)
    return build_report(_run_dir(paths, run_id), paths.reports_dir / run_id)
