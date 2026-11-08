"""Lossless task adapters over the pinned Parquet resources."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import PROJECT_ROOT
from .storage import read_json
from .types import Example

LEGACY_NAMES = {'imdb': ('Negative', 'Positive'), 'rotten_tomatoes': ('Negative', 'Positive'),
                'yelp': ('Negative', 'Positive'), 'ag_news': ('World', 'Sports', 'Business', 'Sci/Tech')}


def _find_names(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, dict):
        names = value.get('names')
        if isinstance(names, list):
            return tuple(names)
        if isinstance(names, dict):
            return tuple(names[str(i)] for i in range(len(names)))
        for child in value.values():
            result = _find_names(child)
            if result:
                return result
    return None


def names_from_parquet(path: Path, dataset: str) -> tuple[str, ...]:
    import pyarrow.parquet as pq
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    features = json.loads(metadata.get(b'huggingface', b'{}')).get('info', {}).get('features', {})
    field = 'labels' if dataset == 'go_emotions' else 'topic' if dataset == 'yahoo_answers_topics' else 'label'
    names = _find_names(features.get(field, {}))
    if names:
        return names
    if dataset in LEGACY_NAMES:
        return LEGACY_NAMES[dataset]
    raise ValueError(f'No authoritative label mapping in {path}')


def examples_from_rows(dataset: str, split: str, rows: Iterable[dict[str, Any]],
                       names: tuple[str, ...], start_row: int = 0) -> Iterator[Example]:
    for number, row in enumerate(rows, start_row):
        fields: dict[str, str] = {}
        if dataset == 'yahoo_answers_topics':
            fields = {key: '' if row.get(key) is None else str(row[key]) for key in ['question_title', 'question_content', 'best_answer']}
            text = ('[TITLE] ' + fields['question_title'] + ' [QUESTION] ' + fields['question_content'] +
                    ' [ANSWER] ' + fields['best_answer']) if any(v.strip() for v in fields.values()) else ''
            value = row['topic']
        else:
            text = '' if row.get('text') is None else str(row['text'])
            value = row['labels'] if dataset == 'go_emotions' else row['label']
        if isinstance(value, str):
            if value not in names:
                raise ValueError(f'{dataset}: unknown label {value}')
            value = names.index(value)
        labels = tuple(sorted(set(int(v) for v in value))) if isinstance(value, (list, tuple)) else (int(value),)
        if not labels or min(labels) < 0 or max(labels) >= len(names):
            raise ValueError(f'{dataset}: labels outside declared target space: {labels}')
        identifier = row.get('id', number)
        yield Example(dataset, f'{dataset}/{split}/{identifier}', text, labels, split, fields=fields)


def iter_examples(dataset: str, lock: dict[str, Any], batch_size: int = 4096) -> Iterator[Example]:
    import pyarrow.parquet as pq
    source = lock['datasets'][dataset]
    names = tuple(source['label_names'])
    offsets: dict[str, int] = {}
    for record in sorted(source['files_resolved'], key=lambda item: item['source_file']):
        name = Path(record['source_file']).name
        split = name.split('-')[0].split('.')[0]
        if split not in {'train', 'validation', 'test'}:
            raise ValueError(f'Unexpected official split: {record["source_file"]}')
        offset = offsets.get(split, 0)
        parquet = pq.ParquetFile(PROJECT_ROOT / record['path'])
        for batch in parquet.iter_batches(batch_size=batch_size):
            rows = batch.to_pylist()
            yield from examples_from_rows(dataset, split, rows, names, offset)
            offset += len(rows)
        offsets[split] = offset


def load_examples(dataset: str, lock: dict[str, Any]) -> tuple[Example, ...]:
    return tuple(iter_examples(dataset, lock))


def label_names(dataset: str) -> tuple[str, ...]:
    if dataset in LEGACY_NAMES:
        return LEGACY_NAMES[dataset]
    lock = read_json(PROJECT_ROOT / 'research-v3/assets/assets.lock.json')
    return tuple(lock['datasets'][dataset]['label_names'])
