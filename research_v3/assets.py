"""Pinned public assets, verified reuse and resumable downloads into new outputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote

from .config import PROJECT_ROOT, canonical_hash, validate_protocol
from .storage import atomic_write_json, bounded_path, hash_file, read_json


def _verify_file(path: Path, spec: dict[str, Any]) -> str:
    actual = hash_file(path)
    expected = spec.get('expected_sha256')
    if expected and actual.lower() != expected.lower():
        raise ValueError(f'Asset SHA256 mismatch: {path}')
    expected_blob = spec.get('expected_git_blob')
    if expected_blob:
        digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_blob:
            raise ValueError(f'Asset Git blob checksum mismatch: {path}')
    if spec.get('expected_bytes') is not None and path.stat().st_size != spec['expected_bytes']:
        raise ValueError(f'Asset byte count mismatch: {path}')
    return actual


def ensure_file(spec: dict[str, Any], destination: Path, offline: bool = False) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.is_file():
        try:
            digest = _verify_file(destination, spec)
            return {**spec, 'path': str(destination), 'bytes': destination.stat().st_size,
                    'sha256': digest, 'cache_reused': True}
        except ValueError:
            if offline:
                raise
    elif offline:
        raise FileNotFoundError(f'Offline asset is missing: {destination}')
    if offline:
        raise ValueError(f'Offline asset cannot be verified: {destination}')
    import requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    last_error = None
    for attempt in range(4):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            if spec.get('expected_bytes') is not None and start > spec['expected_bytes']:
                partial.unlink()
                start = 0
            headers = {'User-Agent': 'tc-explain-v3-preparation/1.0'}
            if start:
                headers['Range'] = f'bytes={start}-'
            url = spec['url'] + ('&' if '?' in spec['url'] else '?') + f'prepare_request={time.time_ns()}'
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                append = start > 0 and response.status_code == 206
                with partial.open('ab' if append else 'wb') as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            try:
                digest = _verify_file(partial, spec)
            except ValueError:
                partial.unlink(missing_ok=True)
                raise
            os.replace(partial, destination)
            return {**spec, 'path': str(destination), 'bytes': destination.stat().st_size,
                    'sha256': digest, 'cache_reused': False}
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'Asset download failed: {destination}: {last_error}') from last_error


def _remote_records(repo: str, revision: str, files: list[str] | None,
                    kind: str, identifier: str, directory: Path) -> list[tuple[dict[str, Any], Path]]:
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    info = (api.dataset_info(repo, revision=revision, files_metadata=True) if kind == 'dataset'
            else api.model_info(repo, revision=revision, files_metadata=True))
    if info.sha != revision:
        raise ValueError(f'Remote revision drift: {repo}: {info.sha}')
    siblings = {f.rfilename: f for f in info.siblings}
    if files is None:
        wanted = {'config.json', 'model.safetensors', 'model.safetensors.index.json',
                  'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                  'vocab.json', 'vocab.txt', 'merges.txt', 'added_tokens.json', 'README.md'}
        files = [name for name in siblings if name in wanted or name.startswith('model-') and name.endswith('.safetensors')]
        if not any(name.endswith('.safetensors') for name in files):
            raise ValueError(f'No safe model weights found: {repo}')
    elif 'README.md' in siblings:
        files = list(files) + ['README.md']
    records = []
    for name in sorted(set(files)):
        if name not in siblings:
            raise FileNotFoundError(f'Pinned source file missing: {repo}/{revision}/{name}')
        target = bounded_path(directory / name, directory)
        sibling = siblings[name]
        lfs = sibling.lfs
        expected_sha = (getattr(lfs, 'sha256', None) or (lfs.get('sha256') if isinstance(lfs, dict) else None)) if lfs else None
        prefix = 'datasets/' if kind == 'dataset' else ''
        record = {'kind': kind, 'dataset_id' if kind == 'dataset' else 'model_id': identifier,
                  'repo': repo, 'revision': revision, 'source_file': name,
                  'url': f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{quote(name, safe="/")}?download=true',
                  'expected_sha256': expected_sha,
                  'expected_git_blob': None if lfs else sibling.blob_id,
                  'expected_bytes': sibling.size}
        records.append((record, target))
    return records


def resolve_assets(p: dict[str, Any], root: Path, offline: bool = False) -> dict[str, Any]:
    validate_protocol(p)
    root = Path(root).resolve()
    lock_path = root / 'assets/assets.lock.json'
    if offline:
        lock = read_json(lock_path)
        if lock['protocol_hash'] != canonical_hash(p):
            raise ValueError('Offline asset lock belongs to a different configuration')
        for record in lock['files']:
            ensure_file({**record, 'expected_sha256': record['sha256']}, PROJECT_ROOT / record['path'], True)
        return lock
    old = read_json(PROJECT_ROOT / 'assets.lock.json')
    old_downloads = {r['path'].replace('\\', '/'): r for r in old['downloads']}
    records, datasets, models = [], {}, {}
    for dataset in p['scope']['single_text_datasets']:
        if dataset in p['sources']:
            source = p['sources'][dataset]
            directory = root / 'assets' / dataset
            records.extend(_remote_records(source['repo'], source['revision'], source['files'], 'dataset', dataset, directory))
            datasets[dataset] = {**source, 'directory': str(directory)}
        else:
            source = old['datasets'][dataset]
            datasets[dataset] = {**source, 'task': 'single_label', 'directory': str(PROJECT_ROOT / 'assets' / dataset)}
            for name in source['files']:
                relative = f'assets/{dataset}/{name}'
                original = old_downloads[relative]
                records.append(({'kind': 'dataset', 'dataset_id': dataset, 'repo': source['repo'],
                                 'revision': source['revision'], 'source_file': name,
                                 'url': original['url'], 'expected_sha256': original['sha256'],
                                 'expected_bytes': original['bytes'], 'legacy_read_only': True}, PROJECT_ROOT / relative))
    source = old['model']
    models['distilbert'] = {**source, 'directory': str(PROJECT_ROOT / 'assets/distilbert')}
    for name in source['files']:
        relative = f'assets/distilbert/{name}'
        original = old_downloads[relative]
        records.append(({'kind': 'model', 'model_id': 'distilbert', 'repo': source['repo'],
                         'revision': source['revision'], 'source_file': name, 'url': original['url'],
                         'expected_sha256': original['sha256'], 'expected_bytes': original['bytes'],
                         'legacy_read_only': True}, PROJECT_ROOT / relative))
    modern_dir = root / 'assets/modernbert'
    modern_revision = p['scope']['modern_reference_revision']
    records.extend(_remote_records(p['scope']['modern_reference'], modern_revision, None, 'model', 'modernbert', modern_dir))
    models['modernbert'] = {'repo': p['scope']['modern_reference'], 'revision': modern_revision, 'directory': str(modern_dir)}
    completed = []
    with ThreadPoolExecutor(max_workers=p.get('preparation', {}).get('download_workers', 4)) as pool:
        jobs = {pool.submit(ensure_file, spec, target, bool(spec.get('legacy_read_only'))): spec for spec, target in records}
        for future in as_completed(jobs):
            record = future.result()
            path = Path(record['path'])
            record['path'] = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
            completed.append(record)
            print(f'ASSET verified {record.get("dataset_id", record.get("model_id"))}/{record["source_file"]}', flush=True)
    from .datasets import names_from_parquet
    for dataset, source in datasets.items():
        paths = [PROJECT_ROOT / r['path'] for r in completed if r.get('dataset_id') == dataset and r['source_file'].endswith('.parquet')]
        source['label_names'] = list(names_from_parquet(paths[0], dataset))
        source['files_resolved'] = [r for r in completed if r.get('dataset_id') == dataset and r['source_file'].endswith('.parquet')]
    completed.sort(key=lambda r: (r['kind'], r.get('dataset_id', r.get('model_id', '')), r['source_file']))
    lock = {'schema_version': '1', 'protocol_hash': canonical_hash(p), 'datasets': datasets, 'models': models,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'files': completed,
            'asset_hash': canonical_hash([(r['repo'], r['revision'], r['source_file'], r['sha256']) for r in completed])}
    atomic_write_json(lock_path, lock, root)
    return lock
