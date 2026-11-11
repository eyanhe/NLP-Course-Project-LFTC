"""Behavior checks for source delivery exclusions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]


class SourcePackageTests(unittest.TestCase):
    def test_package_contains_active_sources_without_local_payloads(self):
        spec = importlib.util.spec_from_file_location("package_source", ROOT / "ops/package_source.py")
        self.assertIsNotNone(spec)
        self.assertTrue((ROOT / "ops/package_source.py").is_file(), "source packaging tool is missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "project"
            for name in ["src/lftc/__init__.py", "configs/paths.yaml", "apps/streamlit_app.py", "ops/run.sh", "docs/guide.md", "docs/archive/2026-11-08.md", "tests/unit/test_example.py", "README.md", "pyproject.toml", ".gitignore", "archive/huge.bin", "artifacts/model.bin", ".venv/python.exe", ".tools/python.exe", "apps/node_modules/a.js", "src/lftc/__pycache__/a.pyc", "docs/.cache/a"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
            target = Path(folder) / "delivery.zip"
            module.package_source(root, target)
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
                self.assertEqual(names, {"src/lftc/__init__.py", "configs/paths.yaml", "apps/streamlit_app.py", "ops/run.sh", "docs/guide.md", "docs/archive/2026-11-08.md", "tests/unit/test_example.py", "README.md", "pyproject.toml", ".gitignore"})
                self.assertEqual(archive.read("docs/guide.md").decode(), "docs/guide.md")


if __name__ == "__main__":
    unittest.main()
