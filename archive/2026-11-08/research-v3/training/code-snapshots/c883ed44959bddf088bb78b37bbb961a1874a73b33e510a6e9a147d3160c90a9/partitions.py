"""Deterministic official-split holdouts and exact/near-duplicate isolation."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import replace
import hashlib
import html
import itertools
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np

from .config import canonical_hash
from .storage import atomic_write_json, hash_file, read_json
from .types import Example, PreparedDataset


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', str(text)))).strip()


def text_digest(text: str) -> bytes:
    return hashlib.sha256(clean_text(text).casefold().encode('utf-8')).digest()


def stable_order(sample_id: str, seed: int) -> bytes:
    return hashlib.sha256(f'{seed}:{sample_id}'.encode('utf-8')).digest()


def _words(text: str) -> list[str]:
    return re.findall(r'\w+', clean_text(text).casefold())


def _shingles(text: str) -> set[bytes]:
    words = _words(text)
    return {' '.join(words[i:i + 5]).encode('utf-8') for i in range(len(words) - 4)}


def _signature_batch(items: list[tuple[int, str]], seed: int) -> list[tuple[int, np.ndarray]]:
    from datasketch import MinHash
    template = MinHash(num_perm=128, seed=seed)
    result = []
    for index, text in items:
        sketch = template.copy()
        sketch.update_batch(list(_shingles(text)))
        result.append((index, sketch.hashvalues.astype(np.uint32)))
    return result


def _find(parent: np.ndarray, index: int) -> int:
    while int(parent[index]) != index:
        parent[index] = parent[int(parent[index])]
        index = int(parent[index])
    return index


def _join(parent: np.ndarray, left: int, right: int) -> bool:
    a, b = _find(parent, left), _find(parent, right)
    if a == b:
        return False
    if a > b:
        a, b = b, a
    parent[b] = a
    return True


def build_duplicate_groups(examples: tuple[Example, ...], seed: int,
                           workspace: Path | None = None, workers: int = 1,
                           progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, str]:
    n = len(examples)
    parent = np.arange(n, dtype=np.int64)
    seen: dict[bytes, int] = {}
    active = []
    identity = hashlib.sha256()
    for i, row in enumerate(examples):
        digest = text_digest(row.text)
        identity.update(row.sample_id.encode('utf-8')); identity.update(digest)
        if digest in seen:
            _join(parent, seen[digest], i)
        else:
            seen[digest] = i
            if len(_words(row.text)) >= 10:
                active.append(i)
        if progress and i and i % 100000 == 0:
            progress({'stage': 'exact_groups', 'processed': i, 'total': n})
    del seen
    source_hash = identity.hexdigest()
    directory = Path(workspace) if workspace is not None else None
    signature_path = directory / 'minhash128.uint32' if directory else None
    meta_path = directory / 'minhash128.meta.json' if directory else None
    use_cache = False
    if meta_path and meta_path.exists() and signature_path.exists():
        metadata = read_json(meta_path)
        use_cache = (metadata.get('complete') is True and metadata.get('source_hash') == source_hash
                     and metadata.get('seed') == seed and metadata.get('rows') == n
                     and hash_file(signature_path) == metadata.get('sha256'))
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        signatures = np.memmap(signature_path, dtype=np.uint32, mode='r+' if use_cache else 'w+', shape=(n, 128))
    else:
        signatures = np.empty((n, 128), dtype=np.uint32)
    if not use_cache:
        batches = ([(i, examples[i].text) for i in active[start:start + 512]] for start in range(0, len(active), 512))
        processed = 0
        last_progress = time.monotonic()
        if workers > 1 and len(active) > 4096:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending = set()
                for batch in itertools.islice(batches, workers * 2):
                    pending.add(pool.submit(_signature_batch, batch, seed))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        values = future.result()
                        for index, signature in values:
                            signatures[index] = signature
                        processed += len(values)
                        next_batch = next(batches, None)
                        if next_batch is not None:
                            pending.add(pool.submit(_signature_batch, next_batch, seed))
                    if progress and time.monotonic() - last_progress >= 5:
                        progress({'stage': 'minhash_signatures', 'processed': processed, 'total': len(active), 'workers': workers})
                        last_progress = time.monotonic()
        else:
            for batch in batches:
                for index, signature in _signature_batch(batch, seed):
                    signatures[index] = signature
        if directory:
            signatures.flush()
            atomic_write_json(meta_path, {'complete': True, 'rows': n, 'seed': seed,
                                         'source_hash': source_hash, 'sha256': hash_file(signature_path)}, directory)
    active_array = np.asarray(active, dtype=np.int64)
    verified_edges, candidates = 0, 0
    visited: set[int] = set()
    from functools import lru_cache

    @lru_cache(maxsize=20000)
    def shingles(index: int) -> frozenset[bytes]:
        return frozenset(_shingles(examples[index].text))

    for band in range(32):
        block = np.ascontiguousarray(signatures[active_array, band * 4:band * 4 + 4])
        keys = block.view('V16').reshape(-1)
        order = np.argsort(keys, kind='stable')
        ordered = keys[order]
        changes = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
        bounds = np.concatenate(([0], changes, [len(order)]))
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop - start < 2:
                continue
            members = active_array[order[start:stop]]
            for a, b in itertools.combinations(members, 2):
                left, right = sorted((int(a), int(b)))
                if _find(parent, left) == _find(parent, right):
                    continue
                key = left * n + right
                if key in visited:
                    continue
                if len(visited) >= 2000000:
                    visited.clear()
                visited.add(key)
                candidates += 1
                sa, sb = shingles(left), shingles(right)
                if min(len(sa), len(sb)) < 0.9 * max(len(sa), len(sb)):
                    continue
                intersection = len(sa & sb)
                if intersection / (len(sa) + len(sb) - intersection) >= 0.9:
                    verified_edges += int(_join(parent, left, right))
        if progress:
            progress({'stage': 'near_duplicate_bands', 'band_completed': band + 1, 'total_bands': 32,
                      'candidate_pairs_checked': candidates, 'verified_component_joins': verified_edges})
    shingles.cache_clear()
    result = {}
    group_names: dict[int, str] = {}
    for i, row in enumerate(examples):
        group = _find(parent, i)
        if group not in group_names:
            group_names[group] = hashlib.sha256(examples[group].sample_id.encode('utf-8')).hexdigest()
        result[row.sample_id] = group_names[group]
    return result


def _purposes(examples: tuple[Example, ...], p: dict[str, Any]) -> dict[str, str]:
    by_dataset: dict[str, list[Example]] = defaultdict(list)
    for row in examples:
        by_dataset[row.dataset_id].append(row)
    result = {}
    seed = p['seeds']['partition']
    for dataset, rows in by_dataset.items():
        for row in rows:
            result[row.sample_id] = 'test' if row.official_split == 'test' else 'train'
        if dataset == 'go_emotions':
            validation = sorted((r for r in rows if r.official_split == 'validation'), key=lambda r: stable_order(r.sample_id, seed))
            cut = int(len(validation) * p['partitions']['official_validation_select_fraction'])
            for i, row in enumerate(validation):
                result[row.sample_id] = 'select' if i < cut else 'discovery'
        elif dataset in {'banking77', 'yahoo_answers_topics'}:
            by_label: dict[int, list[Example]] = defaultdict(list)
            for row in rows:
                if row.official_split == 'train':
                    by_label[row.labels[0]].append(row)
            for label_rows in by_label.values():
                ordered = sorted(label_rows, key=lambda r: stable_order(r.sample_id, seed))
                if dataset == 'banking77':
                    select_n = max(1, int(len(ordered) * p['partitions']['banking_select_fraction']))
                    discovery_n = max(1, int(len(ordered) * p['partitions']['banking_discovery_fraction']))
                    # A tiny class cannot supply two distinct holdouts. Preserve
                    # the rows without duplication; the support audit exposes
                    # the shortfall and downstream conditions become infeasible.
                    select_n = min(select_n, len(ordered))
                    discovery_n = min(discovery_n, len(ordered) - select_n)
                else:
                    select_n = p['partitions']['yahoo_select_per_class']
                    discovery_n = p['partitions']['yahoo_discovery_per_class']
                if select_n + discovery_n > len(ordered):
                    raise ValueError(f'{dataset}: insufficient official training rows for registered holdouts')
                for i, row in enumerate(ordered[:select_n + discovery_n]):
                    result[row.sample_id] = 'select' if i < select_n else 'discovery'
        else:
            identifiers = set(p.get('_legacy_validation_ids', {}).get(dataset, ()))
            if not identifiers:
                raise ValueError(f'{dataset}: historical validation IDs are required')
            validation = sorted((r for r in rows if r.sample_id in identifiers), key=lambda r: stable_order(r.sample_id, seed))
            if len(validation) != len(identifiers):
                raise ValueError(f'{dataset}: historical validation IDs do not match the pinned raw data')
            cut = int(len(validation) * p['partitions']['legacy_select_fraction'])
            for i, row in enumerate(validation):
                result[row.sample_id] = 'select' if i < cut else 'discovery'
            for row in rows:
                if row.official_split == 'validation' and row.sample_id not in identifiers:
                    result[row.sample_id] = 'quarantine'
    return result


def prepare_partitions(examples: tuple[Example, ...], p: dict[str, Any]) -> dict[str, PreparedDataset]:
    identifiers = [row.sample_id for row in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate source sample_id')
    progress = p.get('_progress_callback')
    purposes = _purposes(examples, p)
    groups = build_duplicate_groups(examples, p['seeds']['partition'], p.get('_preparation_workspace'),
                                    p.get('preparation', {}).get('minhash_workers', 1), progress)
    exact_labels: dict[tuple[str, bytes], tuple[int, ...]] = {}
    conflict = set()
    digests = {}
    for row in examples:
        digest = text_digest(row.text)
        digests[row.sample_id] = digest
        key = row.dataset_id, digest
        if key in exact_labels and exact_labels[key] != row.labels:
            conflict.add(key)
        else:
            exact_labels[key] = row.labels
    del exact_labels
    priority = {'quarantine': 0, 'train': 1, 'discovery': 2, 'select': 3, 'test': 4}
    winning: dict[str, int] = {}
    invalid = {}
    for row in examples:
        key = row.dataset_id, digests[row.sample_id]
        if key in conflict:
            invalid[row.sample_id] = 'exact_label_conflict'
        elif not clean_text(row.text):
            invalid[row.sample_id] = 'empty_text'
        else:
            group = groups[row.sample_id]
            winning[group] = max(winning.get(group, 0), priority[purposes[row.sample_id]])
    collected: dict[str, dict[str, list[Example]]] = defaultdict(lambda: defaultdict(list))
    retained_exact = set()
    for row in sorted(examples, key=lambda r: r.sample_id):
        purpose, group = purposes[row.sample_id], groups[row.sample_id]
        reason = invalid.get(row.sample_id)
        if not reason and priority[purpose] < winning.get(group, 0):
            reason = 'duplicate_lower_priority_partition'
        exact_key = row.dataset_id, digests[row.sample_id]
        if not reason and purpose != 'quarantine':
            if exact_key in retained_exact:
                reason = 'redundant_exact_text'
            else:
                retained_exact.add(exact_key)
        if reason or purpose == 'quarantine':
            purpose = 'quarantine'
            fields = {**row.fields, 'quarantine_reason': reason or 'unused_official_validation'}
        else:
            fields = row.fields
        collected[row.dataset_id][purpose].append(replace(row, group_id=group, fields=fields))
    result = {}
    for dataset, partitions in collected.items():
        normalized = {name: tuple(partitions.get(name, ())) for name in ['train', 'select', 'discovery', 'test', 'quarantine']}
        digest = hashlib.sha256()
        for name, rows in normalized.items():
            for row in rows:
                digest.update(f'{name}\0{row.sample_id}\0{row.group_id}\0{row.labels}\n'.encode('utf-8'))
                digest.update(digests[row.sample_id])
        registry = p['_dataset_registry'][dataset]
        result[dataset] = PreparedDataset(dataset, registry['task'], tuple(registry['label_names']), normalized, digest.hexdigest())
    return result
