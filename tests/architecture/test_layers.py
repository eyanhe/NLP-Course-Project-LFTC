"""Prevent the clean core from importing apps, archive or higher layers."""
import ast
from pathlib import Path
import unittest


class LayerTests(unittest.TestCase):
    def test_core_dependencies_follow_layer_direction(self):
        root = Path(__file__).resolve().parents[2] / "src/lftc"
        layers = {"domain": 0, "governance": 1, "config": 1, "data": 2,
                  "features": 2, "models": 2, "evaluation": 2, "training": 3,
                  "reporting": 3, "experiments": 4, "cli": 4, "__main__": 4}
        legacy = {"archive", "research_v3", "pipeline", "extension", "models_extra",
                  "pair_models", "hard_benchmark", "imbalance", "apps"}
        failures = []
        for path in root.rglob("*.py"):
            source = path.relative_to(root).parts[0].removesuffix(".py")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
                for name in names:
                    parts = name.split(".")
                    if parts[0] in legacy:
                        failures.append(f"{path.name}: imports {name}")
                    if len(parts) > 1 and parts[0] == "lftc" and source in layers and parts[1] in layers:
                        if layers[parts[1]] > layers[source]:
                            failures.append(f"{source} imports higher layer {parts[1]}")
        self.assertEqual(failures, [])
