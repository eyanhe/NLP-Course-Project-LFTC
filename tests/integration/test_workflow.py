"""Train, reload, score and report a tiny real local dataset."""
import json
from pathlib import Path
import tempfile
import unittest

import yaml


class WorkflowTests(unittest.TestCase):
    def test_different_holdout_fractions_preserve_prepared_outputs(self):
        from lftc.experiments import prepare
        from lftc.data import load_prepared
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "pyproject.toml").write_text('[project]\nname="lftc"\n', encoding="utf-8")
            configs = root / "configs"
            configs.mkdir()
            (configs / "paths.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            source = root / "artifacts/assets/datasets/toy"
            source.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([
                {"text": f"row {index}", "label": index % 2} for index in range(20)
            ]), source / "train.parquet")
            (configs / "assets.yaml").write_text(yaml.safe_dump({
                "schema_version": 1, "models": [], "datasets": [{
                    "id": "toy", "directory": "datasets/toy", "task": "single_label",
                    "label_count": 2, "files": ["train.parquet"],
                    "repo": "example/toy", "revision": "a" * 40,
                }],
            }), encoding="utf-8")
            outputs = []
            for fraction in (0.1, 0.3):
                path = configs / "prepare.yaml"
                path.write_text(yaml.safe_dump({"schema_version": 1,
                    "dataset": "toy", "model": "tfidf_lr",
                    "data": {"validation_fraction": fraction}}), encoding="utf-8")
                outputs.append(prepare(path, root=root)[0]["directory"])
            self.assertNotEqual(*outputs)
            self.assertEqual([len(load_prepared(path).validation) for path in outputs], [2, 6])

    def test_local_training_prediction_evaluation_and_report(self):
        from lftc.experiments import run, predict, evaluate_run, report_run
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "pyproject.toml").write_text('[project]\nname="lftc"\n', encoding="utf-8")
            configs = root / "configs"
            configs.mkdir()
            (configs / "paths.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            assets = root / "artifacts/assets/datasets/rotten_tomatoes"
            assets.mkdir(parents=True)
            for split, repeats in (("train", 8), ("validation", 2), ("test", 2)):
                rows = [{"text": f"excellent\u0085delightful\u2028good movie {i}", "label": 1}
                        for i in range(repeats)]
                rows += [{"text": f"terrible boring bad movie {i}", "label": 0}
                         for i in range(repeats)]
                pq.write_table(pa.Table.from_pylist(rows), assets / f"{split}.parquet")
            catalog = {
                "schema_version": 1,
                "datasets": [{"id": "rotten_tomatoes", "directory": "datasets/rotten_tomatoes",
                              "repo": "example/reviews", "revision": "a" * 40,
                              "task": "single_label", "label_count": 2,
                              "label_names": ["negative", "positive"],
                              "files": ["train.parquet", "validation.parquet", "test.parquet"]}],
                "models": [],
            }
            (configs / "assets.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")
            config = configs / "tiny.yaml"
            config.write_text(yaml.safe_dump({"schema_version": 1, "name": "tiny",
                "dataset": "rotten_tomatoes", "model": "tfidf_lr", "seed": 42}), encoding="utf-8")
            result = run(config, root=root, run_id="tiny-run")
            self.assertEqual(result["run_id"], "tiny-run")
            self.assertEqual(result["test_rows"], 4)
            self.assertGreater(result["metrics"]["macro_f1"], 0.9)
            predictions = predict("tiny-run", ["delightful good movie", "terrible boring movie"], root=root)
            self.assertEqual([row["prediction"] for row in predictions], [[1], [0]])
            evaluated = evaluate_run("tiny-run", root=root)
            self.assertEqual(evaluated["rows"], 4)
            report = report_run("tiny-run", root=root)
            self.assertTrue(Path(report["markdown"]).is_file())
            self.assertEqual(json.loads((root / "artifacts/runs/tiny-run/run.json").read_text())["dataset"], "rotten_tomatoes")
