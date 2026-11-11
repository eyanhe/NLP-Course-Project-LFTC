"""Bounded local Parquet adapters and deterministic training sampling."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random

from lftc.domain import Example, PreparedDataset, TaskType


def _split(path):
    name = Path(path).stem
    # MNLI has no public labelled test: matched is validation and mismatched
    # is our held-out labelled test. Never use the official unlabelled test.
    if name.startswith('validation_mismatched'):
        return 'test'
    if name.startswith('validation_matched'):
        return 'validation'
    return name.split('-')[0]


def _find_names(value):
    if isinstance(value, dict):
        if isinstance(value.get('names'), list):
            return tuple(value['names'])
        for child in value.values():
            if names := _find_names(child):
                return names
    return ()


def _row_example(asset, row, split, identifier, names):
    pair = None
    if asset.id == 'mnli' or asset.task == TaskType.PAIR_CLASSIFICATION:
        text, pair = row.get('premise'), row.get('hypothesis')
        if not isinstance(pair, str) or not pair.strip():
            return None
    elif asset.id == 'yahoo_answers_topics':
        fields = [row.get(key) or '' for key in ('question_title', 'question_content', 'best_answer')]
        if not any(isinstance(value, str) and value.strip() for value in fields):
            return None
        text = '\n'.join(f'[{key}] {value}' for key, value in zip(('TITLE', 'QUESTION', 'ANSWER'), fields))
    else:
        text = row.get('text')
    if not isinstance(text, str) or not text.strip():
        return None
    value = row.get('labels') if asset.task == TaskType.MULTILABEL else row.get('topic' if asset.id == 'yahoo_answers_topics' else 'label')
    if value is None:
        return None
    values = value if isinstance(value, (list, tuple)) else (value,)
    try:
        labels = tuple(sorted(set(names.index(v) if isinstance(v, str) and v in names else int(v) for v in values)))
    except (ValueError, TypeError, OverflowError):
        return None
    if any(v < 0 or v >= asset.label_count for v in labels):
        return None
    if asset.task != TaskType.MULTILABEL and len(labels) != 1:
        return None
    return Example(identifier, text.strip(), labels, split, pair, asset.id)


def load_dataset(asset, assets_dir, *, limit_per_split=None, validation_fraction=.15, seed=42):
    """Read batches until each requested split limit, without loading whole files.

    A limit selects the physical head of each split and is for smoke tests;
    class-ordered sources can make limited datasets unrepresentative.
    Row IDs encode the source filename and physical row so shard boundaries,
    filtering, and changing a limit never renumber examples. Missing validation
    is drawn only from the loaded training rows (before optional train limiting).
    """
    import pyarrow.parquet as pq
    if limit_per_split is not None and (not isinstance(limit_per_split, int) or limit_per_split < 1):
        raise ValueError('limit_per_split must be a positive integer')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    files = sorted(name for name in asset.files if name.endswith('.parquet'))
    if not files:
        raise ValueError(f'{asset.id}: no Parquet files registered')
    has_validation = any(_split(name) == 'validation' for name in files)
    groups = {name: [] for name in ('train', 'validation', 'test')}
    names = tuple(asset.label_names)
    for name in files:
        split = _split(name)
        if split not in groups:
            continue
        cap = limit_per_split
        if split == 'train' and not has_validation and cap is not None:
            cap = max(2, math.ceil(cap / (1-validation_fraction)))
        if cap is not None and len(groups[split]) >= cap:
            continue
        parquet = pq.ParquetFile(Path(assets_dir) / asset.directory / name)
        if not names:
            metadata = parquet.schema_arrow.metadata or {}
            names = _find_names(json.loads(metadata.get(b'huggingface', b'{}')))
        offset = 0
        for batch in parquet.iter_batches(batch_size=min(4096, max(16, (cap or 4096)))):
            for row in batch.to_pylist():
                example = _row_example(asset, row, split, f'{asset.id}/{name}/{offset}', names)
                offset += 1
                if example is not None:
                    groups[split].append(example)
                if cap is not None and len(groups[split]) >= cap:
                    break
            if cap is not None and len(groups[split]) >= cap:
                break
    if not has_validation and len(groups['train']) > 1:
        indices = list(range(len(groups['train'])))
        random.Random(seed).shuffle(indices)
        n = min(len(indices)-1, max(1, round(len(indices)*validation_fraction)))
        heldout = set(indices[:n])
        groups['validation'] = [replace(row, split='validation') for i, row in enumerate(groups['train']) if i in heldout]
        groups['train'] = [row for i, row in enumerate(groups['train']) if i not in heldout]
    if limit_per_split is not None:
        groups = {key: value[:limit_per_split] for key, value in groups.items()}
    if len(names) != asset.label_count:
        names = tuple(str(i) for i in range(asset.label_count))
    return PreparedDataset(asset.id, TaskType(asset.task), asset.label_count, names,
                           *(tuple(groups[key]) for key in ('train', 'validation', 'test')))


def save_prepared(dataset, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'dataset_id': dataset.dataset_id, 'task': dataset.task.value,
                'label_count': dataset.label_count, 'label_names': dataset.label_names}
    (directory/'dataset.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for split in ('train', 'validation', 'test'):
        with (directory/f'{split}.jsonl').open('w', encoding='utf-8') as stream:
            for example in getattr(dataset, split):
                stream.write(json.dumps(asdict(example), ensure_ascii=False)+'\n')


def load_prepared(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'dataset.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported prepared dataset schema')
    groups = []
    for split in ('train', 'validation', 'test'):
        rows = []
        with (directory/f'{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    row['labels'] = tuple(row['labels'])
                    rows.append(Example(**row))
        groups.append(tuple(rows))
    return PreparedDataset(metadata['dataset_id'], TaskType(metadata['task']), metadata['label_count'],
                           tuple(metadata['label_names']), *groups)


def sample_training(examples, *, mode='all', size=None, k=None, imbalance_ratio=10., seed=42):
    """Select training rows only. Multilabel quotas use the rarest label first."""
    rows = tuple(examples)
    if any(row.split != 'train' for row in rows):
        raise ValueError('Sampling accepts training rows only')
    if mode == 'all':
        return rows
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    if mode in ('scale', 'fraction'):
        if size is None:
            raise ValueError('scale/fraction requires size')
        count = round(len(rows)*size) if isinstance(size, float) and size <= 1 else int(size)
        if count < 1 or count > len(rows):
            raise ValueError('Sample size must be between 1 and the training row count')
        selected = set(order[:count])
    elif mode in ('kshot', 'k_shot', 'long_tail'):
        labels = sorted({label for row in rows for label in row.labels})
        counts = {label: sum(label in row.labels for row in rows) for label in labels}
        if mode != 'long_tail' and (k is None or int(k) < 1):
            raise ValueError('kshot requires positive k')
        if imbalance_ratio < 1:
            raise ValueError('imbalance_ratio must be >= 1')
        quotas = {label: (max(1, round(counts[label]*imbalance_ratio**(-i/max(1, len(labels)-1))))
                          if mode == 'long_tail' else int(k)) for i, label in enumerate(labels)}
        selected = set()
        # Union of label-specific draws: a multilabel row can satisfy several
        # quotas and may cause another label to exceed its minimum quota.
        for label in sorted(labels, key=lambda value: (counts[value], value)):
            present = sum(label in rows[index].labels for index in selected)
            eligible = [index for index in order if index not in selected and label in rows[index].labels]
            selected.update(eligible[:max(0, quotas[label]-present)])
    else:
        raise ValueError(f'Unknown sampling mode: {mode}')
    return tuple(row for i, row in enumerate(rows) if i in selected)
