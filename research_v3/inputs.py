"""A single character-visible prefix shared by every model family."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .partitions import clean_text
from .storage import atomic_write_json, hash_file
from .types import Example, PreparedDataset, VisibleInput


def _visible(row: Example, text: str, offsets: list[tuple[int, int]], limit: int) -> VisibleInput:
    count = min(len(offsets), limit)
    end = len(text) if len(offsets) <= limit else int(offsets[count - 1][1])
    visible = text[:end]
    return VisibleInput(row.sample_id, visible, 0, end, count, len(offsets),
                        hashlib.sha256(visible.encode('utf-8')).hexdigest())


def make_visible_input(example: Example, tokenizer: Any, limit: int = 254) -> VisibleInput:
    if limit < 1 or limit > 254:
        raise ValueError('Content token budget must be between 1 and 254')
    text = clean_text(example.text)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    return _visible(example, text, encoded['offset_mapping'], limit)


def iter_visible_inputs(data: PreparedDataset, tokenizer: Any,
                        batch_size: int = 512) -> Iterator[tuple[str, Example, VisibleInput]]:
    for partition in ['train', 'select', 'discovery', 'test']:
        rows = data.partitions[partition]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [clean_text(row.text) for row in batch]
            encoded = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True,
                                truncation=False, verbose=False)
            for row, text, offsets in zip(batch, texts, encoded['offset_mapping']):
                yield partition, row, _visible(row, text, offsets, 254)


def materialize_inputs(data: PreparedDataset, tokenizer: Any) -> dict[str, VisibleInput]:
    return {value.sample_id: value for _, _, value in iter_visible_inputs(data, tokenizer)}


def write_inputs(data: PreparedDataset, tokenizer: Any, destination: Path,
                 batch_size: int = 512, progress: Any = None) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ('sample_id', pa.string()), ('partition', pa.string()), ('text', pa.string()),
        ('start_char', pa.int32()), ('end_char', pa.int32()), ('content_tokens', pa.int32()),
        ('full_tokens', pa.int32()), ('text_hash', pa.string()), ('normalized_source_hash', pa.string()),
    ], metadata={b'research_v3': json.dumps({'span_basis': 'normalized_text_python_character_offsets',
                                          'main_content_limit': 254, 'special_token_budget': 2}).encode()})
    temporary = destination.with_suffix('.parquet.part')
    counts, lengths = {}, {}
    identity = hashlib.sha256()
    buffer = []
    done = 0
    with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
        for partition, row, value in iter_visible_inputs(data, tokenizer, batch_size):
            identity.update(f'{partition}\0{value.sample_id}\0{value.text_hash}\n'.encode('utf-8'))
            counts[partition] = counts.get(partition, 0) + 1
            lengths.setdefault(partition, []).append(value.full_tokens)
            buffer.append({'sample_id': value.sample_id, 'partition': partition, 'text': value.text,
                           'start_char': value.start_char, 'end_char': value.end_char,
                           'content_tokens': value.content_tokens, 'full_tokens': value.full_tokens,
                           'text_hash': value.text_hash,
                           'normalized_source_hash': hashlib.sha256(clean_text(row.text).encode('utf-8')).hexdigest()})
            done += 1
            if len(buffer) >= 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
                if progress:
                    progress({'stage': 'visible_inputs', 'dataset': data.dataset_id, 'processed': done})
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    temporary.replace(destination)
    partitions = {}
    for name, values in lengths.items():
        array = np.asarray(values, dtype=np.int32)
        partitions[name] = {'n': len(values), 'truncated_n': int((array > 254).sum()),
                            'truncated_fraction': float((array > 254).mean()),
                            'median_full_tokens': float(np.median(array)),
                            'p95_full_tokens': float(np.percentile(array, 95)),
                            'zero_content_tokens': int((array == 0).sum())}
    return {'dataset_id': data.dataset_id, 'rows': done, 'input_hash': identity.hexdigest(),
            'file': str(destination), 'file_sha256': hash_file(destination),
            'span_basis': 'normalized_text_python_character_offsets', 'partitions': partitions,
            'actual_model_training_run': False, 'model_predictions_generated': False}


def audit_inputs(data: PreparedDataset, tokenizer: Any) -> dict[str, Any]:
    values = materialize_inputs(data, tokenizer)
    return {'rows': len(values), 'truncated': sum(v.full_tokens > 254 for v in values.values()),
            'empty': sum(v.content_tokens == 0 for v in values.values())}
