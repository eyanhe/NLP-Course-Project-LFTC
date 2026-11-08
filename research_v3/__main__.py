"""Explicit preparation commands; fitting and official-test predictions are gated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import PROJECT_ROOT, load_protocol, output_root, require_test_enabled, require_training_enabled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['preflight', 'prepare', 'matrix', 'smoke', 'train', 'freeze', 'evaluate',
                 'audit', 'diagnose', 'report', 'verify']:
        item = commands.add_parser(name)
        item.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/research-v3.json')
        item.add_argument('--root', type=Path)
        if name == 'preflight':
            item.add_argument('--out', type=Path)
        if name == 'prepare':
            item.add_argument('--assets-only', action='store_true')
            item.add_argument('--offline', action='store_true')
        if name in {'matrix', 'freeze'}:
            item.add_argument('--stage', default='design')
        if name == 'smoke':
            item.add_argument('--forward-only', action='store_true')
        if name == 'verify':
            item.add_argument('--scope', default='pretraining')
        if name == 'train':
            item.add_argument('--phase', choices=['engineering','anchors','main','mechanisms'], default='main')
            item.add_argument('--authorization', type=Path)
    args = parser.parse_args(argv)
    p = {}
    try:
        p = load_protocol(args.config)
        root = output_root(p)
        if args.root is not None and args.root.resolve() != root:
            raise ValueError('--root must match the isolated root in the verified configuration')
        root.mkdir(parents=True, exist_ok=True)
        if args.command == 'train' and args.authorization is not None:
            from .execution import authorize_phase
            authorize_phase(p, root, args.authorization, args.phase)
        elif args.command in {'train', 'diagnose'} or (args.command == 'smoke' and not args.forward_only):
            require_training_enabled(p)
        if args.command == 'evaluate':
            require_test_enabled(p)
        if args.command == 'preflight':
            from .runtime import probe_environment
            from .storage import bounded_path
            destination = bounded_path(args.out or root / 'environment.json', root)
            result = probe_environment(destination)
            print(json.dumps({'runtime_ready': result['runtime_ready'], 'gpu': result['gpu'],
                              'evidence': str(destination), 'actual_model_training_run': False}, ensure_ascii=False))
            return 0 if result['runtime_ready'] else 1
        if args.command == 'prepare':
            from .assets import resolve_assets
            lock = resolve_assets(p, root, args.offline)
            if args.assets_only:
                print(json.dumps({'assets': len(lock['files']), 'lock': str(root / 'assets/assets.lock.json')}))
                return 0
            from .preparation import prepare_all
            print(json.dumps(prepare_all(p, root, lock), ensure_ascii=False))
            return 0
        if args.command == 'matrix':
            from .matrix import build_saved_matrix
            print(json.dumps(build_saved_matrix(p, root, args.stage), ensure_ascii=False))
            return 0
        if args.command == 'train':
            from .orchestrator import run_training_phase
            result = run_training_phase(p, root, args.phase, args.authorization)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result['complete'] else 1
        if args.command == 'smoke' and args.forward_only:
            from .smoke import run_forward_checks
            result = run_forward_checks(p, root)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result['all_passed'] else 1
        if args.command == 'verify':
            from .verification import verify_pretraining
            report = verify_pretraining(p, root)
            print(json.dumps(report, ensure_ascii=False))
            return 0 if report['complete'] else 1
        raise NotImplementedError(f'{args.command} is not part of the implemented preparation commands')
    except Exception as exc:
        print(json.dumps({'status': 'blocked' if isinstance(exc, PermissionError) else 'failed',
                          'error_type': type(exc).__name__, 'error': str(exc),
                          'actual_model_training_run': bool(p.get('_training_started', False))}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
