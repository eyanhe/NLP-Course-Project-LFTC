"""Exercise configuration through the real CLI without loading model weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "lftc"\n', encoding="utf-8"
        )
        self.write_paths({"schema_version": 1, "artifacts_dir": "artifacts"})
        self.write_assets({
            "schema_version": 1,
            "datasets": [{
                "id": "sample", "directory": "datasets/sample",
                "task": "single_label", "label_count": 2,
                "repo": "example/sample", "revision": "a" * 40,
                "files": ["train.parquet"],
            }],
            "models": [{
                "id": "base", "directory": "models/base",
                "repo": "example/base", "revision": "b" * 40,
                "files": ["config.json"],
            }],
        })

    def write_paths(self, data):
        (self.config_dir / "paths.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8"
        )

    def write_assets(self, data):
        (self.config_dir / "assets.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8"
        )

    def run_cli(self, *args, overrides=None):
        env = os.environ.copy()
        for key in ("LFTC_ROOT", "LFTC_CONFIG_DIR", "LFTC_ARTIFACTS_DIR"):
            env.pop(key, None)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        env["PYTHONIOENCODING"] = "utf-8"
        env.update(overrides or {})
        return subprocess.run(
            [sys.executable, "-B", "-m", "lftc", "--root", str(self.root), *args],
            cwd=self.root.parent, env=env, capture_output=True,
            text=True, encoding="utf-8",
        )

    def test_env_uses_explicit_root_from_another_directory(self):
        result = self.run_cli("env")
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)["paths"]
        self.assertEqual(paths["root"], str(self.root))
        self.assertEqual(paths["assets_dir"], str(self.root / "artifacts/assets"))

    def test_artifacts_override_redirects_all_product_directories(self):
        external = self.root.parent / "external-artifacts"
        result = self.run_cli("env", overrides={"LFTC_ARTIFACTS_DIR": str(external)})
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)["paths"]
        self.assertEqual(paths["assets_dir"], str(external / "assets"))
        self.assertEqual(paths["runs_dir"], str(external / "runs"))
        self.assertEqual(paths["reports_dir"], str(external / "reports"))

    def test_config_override_selects_paths_configuration(self):
        custom = self.root / "custom-configs"
        custom.mkdir()
        (custom / "paths.yaml").write_text(
            "schema_version: 1\nartifacts_dir: products\n", encoding="utf-8"
        )
        result = self.run_cli("env", overrides={"LFTC_CONFIG_DIR": str(custom)})
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = json.loads(result.stdout)["paths"]
        self.assertEqual(paths["config_dir"], str(custom))
        self.assertEqual(paths["assets_dir"], str(self.root / "products/assets"))

    def test_assets_list_does_not_require_local_data_or_model_weights(self):
        result = self.run_cli("assets", "list")
        self.assertEqual(result.returncode, 0, result.stderr)
        assets = json.loads(result.stdout)
        self.assertEqual(assets["datasets"][0]["id"], "sample")
        self.assertEqual(assets["datasets"][0]["label_count"], 2)
        self.assertEqual(assets["models"][0]["id"], "base")

    def test_missing_catalog_returns_structured_error(self):
        (self.config_dir / "assets.yaml").unlink()
        result = self.run_cli("assets", "list")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr)["error"], "ConfigError")
        self.assertEqual(result.stdout, "")

    def test_invalid_task_is_reported_as_configuration_error(self):
        path = self.config_dir / "assets.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["datasets"][0]["task"] = "unknown"
        self.write_assets(data)
        result = self.run_cli("assets", "list")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr)["error"], "ConfigError")

    def test_relative_product_path_cannot_escape_artifacts_directory(self):
        self.write_paths({"schema_version": 1, "assets_dir": "../outside"})
        result = self.run_cli("env")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr)["error"], "PathBoundaryError")

    def test_unsupported_schema_returns_configuration_error(self):
        self.write_paths({"schema_version": 2})
        result = self.run_cli("env")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr)["error"], "ConfigError")
