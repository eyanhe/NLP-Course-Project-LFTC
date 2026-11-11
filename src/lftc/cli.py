"""Unified CLI for experiments, evaluation, reports and the application."""
import argparse
from dataclasses import asdict
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from lftc.config import load_assets_config, load_paths_config
from lftc.domain import LftcError
from lftc.governance import bounded_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lftc")
    parser.add_argument("--root", type=Path, help="Project root (otherwise cwd/LFTC_ROOT)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="Show workspace paths and available optional packages")
    assets = commands.add_parser("assets", help="Inspect registered datasets and base models")
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    listing = asset_commands.add_parser("list", help="List asset metadata without reading data")
    listing.add_argument("--config", type=Path, help="Alternative asset catalog")
    for name in ("prepare", "matrix", "train", "run"):
        command = commands.add_parser(name, help=f"{name.capitalize()} an experiment recipe or grid")
        command.add_argument("--config", type=Path, required=True)
        if name in {"train", "run"}:
            command.add_argument("--run-id", help="Name for a single saved run")
        if name == "matrix":
            command.add_argument("--output", type=Path, help="Also save the expanded grid as JSON")
    for name in ("evaluate", "diagnose"):
        command = commands.add_parser(name, help=f"{name.capitalize()} saved test predictions")
        command.add_argument("--run-id", required=True)
    prediction = commands.add_parser("predict", help="Predict texts with a newly saved model")
    prediction.add_argument("--run-id", required=True)
    prediction.add_argument("--text", action="append", required=True)
    prediction.add_argument("--text-pair", action="append")
    comparison = commands.add_parser("compare", help="Paired bootstrap and McNemar comparison")
    comparison.add_argument("--run-a", required=True)
    comparison.add_argument("--run-b", required=True)
    comparison.add_argument("--resamples", type=int, default=1000)
    report = commands.add_parser("report", help="Export a new run or the historical research")
    report_scope = report.add_mutually_exclusive_group(required=True)
    report_scope.add_argument("--run-id")
    report_scope.add_argument("--history", action="store_true")
    report.add_argument("--output", type=Path)
    commands.add_parser("history", help="Summarize the completed historical research")
    runs = commands.add_parser("runs", help="List new experiment runs")
    runs.add_argument("action", nargs="?", choices=["list"], default="list")
    app = commands.add_parser("app", help="Start the unified Streamlit application")
    app.add_argument("--port", type=int, default=8501)
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--headless", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = load_paths_config(args.root)
        if args.command == "env":
            output = {
                "paths": {key: str(value) for key, value in asdict(paths).items()},
                "python": platform.python_version(),
                "optional_dependencies": {
                    name: find_spec(name) is not None
                    for name in ("numpy", "pandas", "torch", "transformers", "streamlit")
                },
            }
        elif args.command == "assets":
            catalog = load_assets_config(args.config or paths.config_dir / "assets.yaml")
            output = {
                group: [
                    {**asdict(asset), "path": str(bounded_path(asset.directory, paths.assets_dir))}
                    for asset in getattr(catalog, group)
                ]
                for group in ("datasets", "models")
            }
        elif args.command in {"prepare", "matrix", "train", "run", "predict", "evaluate",
                              "diagnose", "compare"}:
            from lftc import experiments
            if args.command == "prepare":
                output = experiments.prepare(args.config, root=paths.root)
            elif args.command == "matrix":
                output = experiments.matrix(args.config, root=paths.root, output=args.output)
            elif args.command in {"train", "run"}:
                output = experiments.run(args.config, root=paths.root, run_id=args.run_id)
            elif args.command == "predict":
                output = experiments.predict(args.run_id, args.text, args.text_pair, root=paths.root)
            elif args.command == "evaluate":
                output = experiments.evaluate_run(args.run_id, root=paths.root)
            elif args.command == "diagnose":
                output = experiments.diagnose_run(args.run_id, root=paths.root)
            else:
                output = experiments.compare_runs(args.run_a, args.run_b, root=paths.root,
                                                   n_resamples=args.resamples)
        elif args.command in {"history", "report", "runs"}:
            from lftc import reporting
            if args.command == "runs":
                output = reporting.list_runs(paths)
            elif args.command == "history":
                data = reporting.load_history(paths.root)
                output = {"historical": True, "counts": data["counts"],
                          "q_records": len(data["q_records"]),
                          "reports_available": len(data.get("downloads", []))}
            elif args.history:
                output = reporting.build_history_report(
                    paths.root, args.output or paths.reports_dir / "historical-research")
            elif args.output:
                output = reporting.build_report(
                    bounded_path(args.run_id, paths.runs_dir), args.output)
            else:
                from lftc.experiments import report_run
                output = report_run(args.run_id, root=paths.root)
        else:
            if not 1 <= args.port <= 65535:
                parser.error("--port must be between 1 and 65535")
            script = paths.root / "apps/streamlit_app.py"
            return subprocess.call([
                sys.executable, "-m", "streamlit", "run", str(script),
                "--server.address", args.host, "--server.port", str(args.port),
                "--server.headless", str(args.headless).lower(),
                "--browser.gatherUsageStats", "false",
            ], cwd=paths.root, env={**os.environ, "LFTC_ROOT": str(paths.root)})
        print(json.dumps(output, indent=2))
        return 0
    except (LftcError, OSError, ValueError, ImportError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
