"""Reproducible, offline-after-download sentiment experiments.

Stages: prepare -> baselines -> smoke -> transformers -> freeze -> evaluate -> report.
Test labels never select checkpoints or hyperparameters.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import time
import warnings

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

NAMES = ['imdb', 'rotten_tomatoes', 'yelp']
MODEL_NAMES = ['logistic_regression', 'linear_svm', 'distilbert']
CONFIG = json.loads((ROOT / 'config.json').read_text('utf-8'))

def utcnow(): return datetime.now(timezone.utc).isoformat()
def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    temp.replace(path)
def read_json(path): return json.loads(Path(path).read_text('utf-8'))
def log(message): print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)
def clean_text(text):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', str(text)))).strip()
def text_hash(text): return hashlib.sha256(clean_text(text).casefold().encode()).hexdigest()
def scores(y, pred):
    return {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, pred)),
            'classification_report': classification_report(y, pred, labels=[0, 1],
                target_names=['negative', 'positive'], output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
def load_split(dataset, split):
    path = ROOT / 'data' / dataset / f'{split}.jsonl'
    expected = read_json(ROOT / 'data' / 'audit.json')['manifest_sha256'][f'{dataset}/{split}']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Prepared data changed after freezing: {dataset}/{split}')
    return pd.read_json(path, lines=True)
def split_hashes(): return read_json(ROOT / 'data' / 'audit.json')['manifest_sha256']
def provenance():
    return {'config_sha256': canonical_hash(CONFIG), 'manifest_sha256': split_hashes(),
            'model_revision': read_json(ROOT / 'assets.lock.json')['model']['revision']}
def assert_provenance(metadata):
    current = provenance()
    for key in current:
        if metadata.get(key) != current[key]: raise RuntimeError(f'Stale output: {key} differs. Use a new project copy for a new protocol.')
def hardware():
    import torch
    return {'platform': platform.platform(), 'processor': platform.processor(),
            'logical_cpus': os.cpu_count(), 'cpu_threads': CONFIG['cpu_threads'],
            'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}

def balanced_sample(frame, n, rng):
    per_class = n // 2
    available = min(int((frame.label == v).sum()) for v in [0, 1])
    per_class = min(per_class, available)
    selected = np.concatenate([rng.choice(frame.index[frame.label == v], per_class, replace=False) for v in [0, 1]])
    return frame.loc[rng.permutation(selected)].copy()

def prepare():
    if (ROOT / 'data' / 'audit.json').exists():
        audit = read_json(ROOT / 'data' / 'audit.json')
        if audit['config_sha256'] != canonical_hash(CONFIG): raise RuntimeError('Existing data belongs to another configuration')
        for source in NAMES:
            for split in ['train','validation','test']: load_split(source,split)
        log('Prepared data already exists; reusing frozen manifests.'); return
    if (ROOT / 'results' / 'protocol.freeze.json').exists(): raise RuntimeError('Evaluation protocol is frozen; do not alter the data.')
    lock = read_json(ROOT / 'assets.lock.json')
    frames = []
    original = {}
    for order, name in enumerate(NAMES):
        for split in ['train', 'validation', 'test']:
            files = [ROOT / 'assets' / name / f for f in lock['datasets'][name]['files']
                     if Path(f).name.startswith(split)]
            if not files: continue
            frame = pd.concat([pd.read_parquet(f, columns=['text', 'label']) for f in files], ignore_index=True)
            if not set(frame.label.unique()).issubset({0, 1}): raise ValueError(f'Unexpected labels in {name}/{split}')
            frame['dataset'] = name; frame['official_split'] = split; frame['dataset_order'] = order
            frame['sample_id'] = [f'{name}/{split}/{i}' for i in range(len(frame))]
            frame['text'] = frame.text.map(clean_text)
            frame['text_hash'] = frame.text.map(text_hash)
            frame['priority'] = {'test': 0, 'validation': 1, 'train': 2}[split]
            original[f'{name}/{split}'] = len(frame)
            frames.append(frame)
        log(f'Loaded {name}')
    all_rows = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    conflicts = all_rows.groupby('text_hash').label.nunique()
    conflict_hashes = set(conflicts[conflicts > 1].index)
    eligible = all_rows[~all_rows.text_hash.isin(conflict_hashes) & (all_rows.text.str.len() > 0)]
    eligible = eligible.sort_values(['priority', 'dataset_order', 'sample_id'], kind='stable').drop_duplicates('text_hash', keep='first')
    audit = {'created_utc': utcnow(), 'config_sha256': canonical_hash(CONFIG), 'official_counts': original,
             'original_rows': len(all_rows), 'eligible_rows': len(eligible),
             'conflicting_unique_hashes': len(conflict_hashes),
             'dedup_policy': 'Normalized casefold SHA256; test before official validation before train; dataset order imdb,rotten_tomatoes,yelp; conflicting labels removed globally.',
             'datasets': {}, 'manifest_sha256': {}}
    rng = np.random.default_rng(CONFIG['sample_seed'])
    manifests = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'assets' / 'distilbert', local_files_only=True)
    selected_frames = []
    for name in NAMES:
        subset = eligible[eligible.dataset == name]
        train_pool = subset[subset.official_split == 'train']
        if name == 'rotten_tomatoes':
            train = balanced_sample(train_pool, CONFIG['train_size'], rng)
            val = balanced_sample(subset[subset.official_split == 'validation'], CONFIG['validation_size'], rng)
        else:
            val = balanced_sample(train_pool, CONFIG['validation_size'], rng)
            train = balanced_sample(train_pool.drop(index=val.index), CONFIG['train_size'], rng)
        test = balanced_sample(subset[subset.official_split == 'test'], CONFIG['test_size'], rng)
        audit['datasets'][name] = {}
        for split, frame in [('train', train), ('validation', val), ('test', test)]:
            frame = frame[['sample_id', 'dataset', 'official_split', 'text', 'label', 'text_hash']].copy()
            frame['split'] = split
            folder = ROOT / 'data' / name; folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{split}.jsonl'
            frame.to_json(path, orient='records', lines=True, force_ascii=False)
            lengths = []
            for start in range(0, len(frame), 256):
                lengths.extend(len(x) for x in tokenizer(frame.text.iloc[start:start+256].tolist(), truncation=False)['input_ids'])
            words = frame.text.str.split().str.len()
            audit['datasets'][name][split] = {'n': len(frame), 'negative': int((frame.label == 0).sum()),
                'positive': int((frame.label == 1).sum()), 'mean_words': float(words.mean()),
                'median_tokens': float(np.median(lengths)), 'p95_tokens': float(np.percentile(lengths, 95)),
                'truncated_fraction': float(np.mean(np.array(lengths) > CONFIG['max_length']))}
            audit['manifest_sha256'][f'{name}/{split}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifests.append(frame.drop(columns='text')); selected_frames.append(frame)
        log(f'Prepared {name}: train={len(train)}, val={len(val)}, test={len(test)}')
    manifest = pd.concat(manifests, ignore_index=True)
    if manifest.text_hash.duplicated().any(): raise AssertionError('Global exact-duplicate leak')
    manifest.to_csv(ROOT / 'data' / 'manifest.csv', index=False)
    # Exact short-snippet containment in sampled long reviews: deterministic auxiliary audit.
    short = pd.concat(selected_frames, ignore_index=True)
    rt = short[short.dataset == 'rotten_tomatoes']
    probes = rt[rt.text.str.len() >= 50].sample(min(100, int((rt.text.str.len() >= 50).sum())), random_state=2026)
    long_reviews = short[short.dataset.isin(['imdb', 'yelp'])]
    haystack = '\n'.join(long_reviews.text.str.casefold())
    found = probes[probes.text.str.casefold().map(lambda t: t in haystack)]
    audit['containment_probe'] = {'probe_count': len(probes), 'matching_count': len(found),
                                 'sample_ids': found.sample_id.tolist(),
                                 'limitation': 'Only 100 sampled short snippets >=50 characters; no semantic near-duplicate or movie/author grouping guarantee.'}
    audit['all_selected_hashes_unique'] = True
    write_json(ROOT / 'data' / 'audit.json', audit)
    log('Data manifests frozen; no global exact-text overlap.')

def model_dir(source, model, seed=None):
    return ROOT / 'models' / source / (f'{model}_seed{seed}' if seed is not None else model)

def baselines():
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('error', category=ConvergenceWarning)
    for source in NAMES:
        train, val = load_split(source, 'train'), load_split(source, 'validation')
        for name in MODEL_NAMES[:2]:
            dest = model_dir(source, name)
            if (dest / 'metadata.json').exists():
                assert_provenance(read_json(dest / 'metadata.json')); log(f'Reuse {source}/{name}'); continue
            start = time.perf_counter(); candidates = []; best = -1.; winner = None
            for ngram in CONFIG['tfidf_ngrams']:
                for c in CONFIG['linear_C']:
                    classifier = LogisticRegression(C=c, solver='lbfgs', max_iter=3000) if name == 'logistic_regression' else LinearSVC(C=c, random_state=42, max_iter=10000)
                    model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=tuple(ngram),
                        max_features=CONFIG['tfidf_max_features'], min_df=CONFIG['tfidf_min_df'], sublinear_tf=True)), ('classifier', classifier)])
                    tick = time.perf_counter()
                    with threadpool_limits(limits=CONFIG['cpu_threads']): model.fit(train.text, train.label)
                    elapsed = time.perf_counter() - tick
                    metric = scores(val.label, model.predict(val.text))
                    candidates.append({'ngram': ngram, 'C': c, 'fit_seconds': elapsed, 'validation': metric})
                    log(f'{source}/{name}: ngram={ngram}, C={c}, val F1={metric["macro_f1"]:.4f}')
                    if metric['macro_f1'] > best:
                        best = metric['macro_f1']; winner = model; selected = candidates[-1]
            dest.mkdir(parents=True, exist_ok=True); joblib.dump(winner, dest / 'model.joblib')
            metadata = {**provenance(), 'created_utc': utcnow(), 'source': source, 'model': name,
                'seed': None, 'score_type': 'probability' if name == 'logistic_regression' else 'margin',
                'selected': selected, 'candidates': candidates, 'search_wall_seconds': time.perf_counter()-start,
                'model_bytes': (dest / 'model.joblib').stat().st_size,
                'tie_break': 'First candidate in fixed ngram then C order; validation Macro-F1 only.'}
            write_json(dest / 'metadata.json', metadata)

def torch_setup(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(CONFIG['cpu_threads'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode(frame, tokenizer):
    cached = tokenizer(frame.text.tolist(), truncation=True, max_length=CONFIG['max_length'], padding=False)
    return [{**{k: v[i] for k, v in cached.items()}, 'labels': int(frame.label.iloc[i])} for i in range(len(frame))]

def loader(rows, tokenizer, batch_size, shuffle=False, seed=42):
    import torch
    from transformers import DataCollatorWithPadding
    return torch.utils.data.DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8), num_workers=0,
        generator=torch.Generator().manual_seed(seed), pin_memory=torch.cuda.is_available())

def torch_predict(model, tokenizer, frame, device, batch_size=32):
    import torch
    rows = encode(frame, tokenizer); predictions = []; probabilities = []
    model.eval()
    with torch.inference_mode():
        for batch in loader(rows, tokenizer, batch_size):
            batch.pop('labels')
            inputs = {k: v.to(device) for k, v in batch.items()}
            output = model(**inputs).logits.float().softmax(-1)
            predictions.extend(output.argmax(-1).cpu().tolist())
            probabilities.extend(output[:, 1].cpu().tolist())
    return np.asarray(predictions), np.asarray(probabilities)

def train_transformer(source, lr, seed, dest, smoke=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    metadata_path = dest / 'metadata.json'
    if metadata_path.exists():
        metadata = read_json(metadata_path); assert_provenance(metadata); return metadata
    device = torch_setup(seed)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'assets' / 'distilbert', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / 'assets' / 'distilbert',
        local_files_only=True, num_labels=2, attn_implementation='eager',
        id2label={0: 'NEGATIVE', 1: 'POSITIVE'}, label2id={'NEGATIVE': 0, 'POSITIVE': 1}).to(device)
    train, val = load_split(source, 'train'), load_split(source, 'validation')
    if smoke: train, val = train.iloc[:200], val.iloc[:100]
    prep_start = time.perf_counter()
    rows = encode(train, tokenizer)
    train_loader = loader(rows, tokenizer, CONFIG['batch_size'], shuffle=True, seed=seed)
    epochs = 1 if smoke else CONFIG['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG['weight_decay'])
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * CONFIG['warmup_ratio']), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    dest.mkdir(parents=True, exist_ok=True)
    history = []; best = -1.; best_epoch = 0; start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                loss = model(**batch).loss
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= old_scale: scheduler.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0: log(f'{source} seed={seed} lr={lr:g} epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(losses[-100:]):.4f}')
        pred, _ = torch_predict(model, tokenizer, val, device)
        metric = scores(val.label, pred)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        'epoch_wall_seconds': time.perf_counter()-epoch_start, 'validation': metric})
        log(f'{source} seed={seed} lr={lr:g} epoch={epoch}: loss={np.mean(losses):.4f}, val F1={metric["macro_f1"]:.4f}')
        if metric['macro_f1'] > best:
            best = metric['macro_f1']; best_epoch = epoch
            model.save_pretrained(dest, safe_serialization=True); tokenizer.save_pretrained(dest)
    metadata = {**provenance(), 'created_utc': utcnow(), 'source': source, 'model': 'distilbert',
        'seed': seed, 'learning_rate': lr, 'max_length': CONFIG['max_length'], 'batch_size': CONFIG['batch_size'],
        'train_n': len(train), 'validation_n': len(val), 'epochs_ran': epochs, 'best_epoch': best_epoch,
        'best_validation_macro_f1': best, 'history': history,
        'preprocessing_seconds': start-prep_start, 'fit_wall_seconds': time.perf_counter()-start,
        'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else None,
        'device': str(device), 'score_type': 'probability', 'smoke_only': smoke, 'hardware': hardware()}
    metadata['model_bytes'] = sum(p.stat().st_size for p in dest.iterdir() if p.is_file())
    write_json(metadata_path, metadata)
    del model, optimizer, scheduler, scaler; gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return metadata

def transformers_stage(smoke=False):
    if smoke:
        result = train_transformer('imdb', CONFIG['learning_rates'][0], 42, ROOT / '.cache' / 'smoke', smoke=True)
        write_json(ROOT / 'results' / 'smoke.json', result)
        log(f'Smoke completed in {result["fit_wall_seconds"]:.1f}s. Full workload has at most 13,500 optimizer steps.'); return
    selections = {}
    for source in NAMES:
        candidates = []
        for lr in CONFIG['learning_rates']:
            dest = ROOT / '.cache' / 'candidates' / source / f'lr{lr:g}_seed42'
            result = train_transformer(source, lr, 42, dest)
            candidates.append((result, dest))
        # Fixed tie-break favors first (lower) learning rate; test results never inspected.
        best, candidate = max(candidates, key=lambda pair: pair[0]['best_validation_macro_f1'])
        dest = model_dir(source, 'distilbert', 42)
        if not dest.exists(): shutil.copytree(candidate, dest)
        selections[source] = {'learning_rate': best['learning_rate'],
            'candidate_metadata': [r for r, _ in candidates], 'selection': 'highest validation Macro-F1; lower LR wins ties'}
        for seed in CONFIG['training_seeds'][1:]:
            train_transformer(source, best['learning_rate'], seed, model_dir(source, 'distilbert', seed))
    write_json(ROOT / 'results' / 'transformer_selection.json', {**provenance(), 'selections': selections})

def freeze():
    artifacts = []
    for source in NAMES:
        for name in MODEL_NAMES:
            for seed in (CONFIG['training_seeds'] if name == 'distilbert' else [None]):
                dest = model_dir(source, name, seed)
                metadata = read_json(dest / 'metadata.json'); assert_provenance(metadata)
                if metadata.get('smoke_only'): raise RuntimeError('Smoke checkpoint cannot enter final evaluation')
                weights = dest / ('model.safetensors' if name == 'distilbert' else 'model.joblib')
                artifacts.append({'source': source, 'model': name, 'seed': seed, 'path': str(dest.relative_to(ROOT)),
                    'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest()})
    frozen = {**provenance(), 'frozen_utc': utcnow(), 'optional_learning_curve': False,
              'artifacts': artifacts, 'note': 'No test-based selection; no optional experiment. Freeze created before test metrics.'}
    path = ROOT / 'results' / 'protocol.freeze.json'
    if path.exists():
        old = read_json(path); assert_provenance(old)
        if old['artifacts'] != artifacts: raise RuntimeError('Frozen artifacts have changed')
    else: write_json(path, frozen)
    log('Protocol and 15 final artifacts frozen before test evaluation.')

def load_model(source, name, seed=42, device='cpu'):
    dest = model_dir(source, name, seed if name == 'distilbert' else None)
    if name != 'distilbert': return joblib.load(dest / 'model.joblib'), None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(dest, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(dest, local_files_only=True, attn_implementation='eager').to(device).eval()
    return model, tokenizer

def predict_loaded(model, tokenizer, texts, name, device='cpu'):
    texts = [clean_text(t) for t in texts]
    if name == 'distilbert':
        frame = pd.DataFrame({'text': texts, 'label': [0]*len(texts)})
        return torch_predict(model, tokenizer, frame, device)
    pred = model.predict(texts)
    score = model.decision_function(texts) if name == 'linear_svm' else model.predict_proba(texts)[:, 1]
    return np.asarray(pred), np.asarray(score)

def fast_macro_f1(y, pred):
    tp = np.sum((y == 1) & (pred == 1), axis=-1)
    tn = np.sum((y == 0) & (pred == 0), axis=-1)
    fp = np.sum((y == 0) & (pred == 1), axis=-1)
    fn = np.sum((y == 1) & (pred == 0), axis=-1)
    return .5 * (2*tp/np.maximum(2*tp+fp+fn, 1) + 2*tn/np.maximum(2*tn+fp+fn, 1))

def paired_bootstrap(y, a, b, repetitions=1000):
    rng = np.random.default_rng(2026)
    ids = np.concatenate([rng.choice(np.where(y == label)[0], size=(repetitions, int(np.sum(y == label))), replace=True)
                          for label in [0, 1]], axis=1)
    delta = fast_macro_f1(y[ids], b[ids]) - fast_macro_f1(y[ids], a[ids])
    return {'comparison': 'DistilBERT seed42 minus SVM', 'delta_macro_f1': float(fast_macro_f1(y,b)-fast_macro_f1(y,a)),
            'ci95_low': float(np.quantile(delta, .025)), 'ci95_high': float(np.quantile(delta, .975)),
            'replicates': repetitions, 'interpretation': 'Paired stratified percentile bootstrap; conditional on two fixed trained models, not training uncertainty.'}

def evaluate():
    import torch
    frozen = read_json(ROOT / 'results' / 'protocol.freeze.json'); assert_provenance(frozen)
    completed = ROOT / 'results' / 'evaluation.json'
    if completed.exists() and all((ROOT / 'results' / f).exists() for f in ['predictions.csv','metrics.csv','bootstrap.json']):
        assert_provenance(read_json(completed)); log('Reusing completed frozen evaluation.'); return
    torch.set_num_threads(CONFIG['cpu_threads'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test = {name: load_split(name, 'test') for name in NAMES}
    rows = []; frames = []
    for artifact in frozen['artifacts']:
        source, name, seed = artifact['source'], artifact['model'], artifact['seed']
        checkpoint = ROOT / artifact['path']
        weights = checkpoint / ('model.safetensors' if name == 'distilbert' else 'model.joblib')
        if hashlib.sha256(weights.read_bytes()).hexdigest() != artifact['weights_sha256']: raise RuntimeError('Frozen checkpoint changed')
        model, tok = load_model(source, name, seed, device)
        for target, frame in test.items():
            with threadpool_limits(limits=CONFIG['cpu_threads']):
                pred, score = predict_loaded(model, tok, frame.text.tolist(), name, device)
            metric = scores(frame.label, pred)
            rows.append({'source': source, 'target': target, 'model': name, 'seed': seed,
                         'test_n': len(frame), **metric})
            frames.append(pd.DataFrame({'sample_id': frame.sample_id, 'source': source, 'target': target,
                'model': name, 'seed': seed, 'label': frame.label, 'prediction': pred, 'score': score,
                'score_type': 'margin' if name == 'linear_svm' else 'positive_probability'}))
            log(f'{source} -> {target} {name} seed={seed}: F1={metric["macro_f1"]:.4f}')
        del model, tok; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
    write_json(ROOT / 'results' / 'evaluation.json', {**provenance(), 'created_utc': utcnow(), 'rows': rows})
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(ROOT / 'results' / 'predictions.csv', index=False)
    scalar = pd.DataFrame([{k: v for k,v in row.items() if k not in ['classification_report','confusion_matrix']} for row in rows])
    scalar.to_csv(ROOT / 'results' / 'metrics.csv', index=False)
    bootstraps = {}
    errors = []
    for target in NAMES:
        frame = test[target].set_index('sample_id')
        within = predictions[(predictions.source == target) & (predictions.target == target)]
        svm = within[within.model == 'linear_svm'].set_index('sample_id').reindex(frame.index)
        bert = within[(within.model == 'distilbert') & (within.seed == 42)].set_index('sample_id').reindex(frame.index)
        bootstraps[target] = paired_bootstrap(frame.label.to_numpy(), svm.prediction.to_numpy(), bert.prediction.to_numpy(), CONFIG['bootstrap_replicates'])
        pivot = within[(within.model != 'distilbert') | (within.seed == 42)].pivot(index='sample_id', columns='model', values='prediction').reindex(frame.index)
        incorrect = pivot.ne(frame.label, axis=0).any(axis=1)
        candidate = frame[incorrect].copy()
        selected = []
        for label in [0, 1]:
            available = candidate[candidate.label == label]
            selected.extend(available.sample(min(10, len(available)), random_state=2026).index)
        for idx in selected:
            record = {'dataset': target, 'sample_id': idx, 'text': frame.loc[idx,'text'], 'label': int(frame.loc[idx,'label']),
                      **{name: int(pivot.loc[idx,name]) for name in MODEL_NAMES},
                      'human_categories': '', 'human_explanation': '', 'reviewer_1': '', 'reviewer_2': '',
                      'review_status': 'unreviewed'}
            errors.append(record)
    write_json(ROOT / 'results' / 'bootstrap.json', bootstraps)
    # Preserve genuine human work on a resumed evaluation.
    error_path = ROOT / 'results' / 'error_review.csv'
    if not error_path.exists(): pd.DataFrame(errors).to_csv(error_path, index=False, encoding='utf-8-sig')
    log('Test evaluation complete. Human error-review fields deliberately unfilled.')

def benchmark():
    import torch
    assert_provenance(read_json(ROOT / 'results' / 'protocol.freeze.json'))
    completed = ROOT / 'results' / 'benchmark.json'
    if completed.exists():
        assert_provenance(read_json(completed)); log('Reusing completed CPU benchmark.'); return
    torch.set_num_threads(CONFIG['cpu_threads'])
    # Same fixed 200 texts for every artifact, all three data sources represented.
    pooled = pd.concat([load_split(name, 'test') for name in NAMES], ignore_index=True)
    sample = pooled.sample(CONFIG['benchmark_samples'], random_state=2026)
    results = []
    for source in NAMES:
        for name in MODEL_NAMES:
            model, tok = load_model(source, name, 42, 'cpu')
            with threadpool_limits(limits=CONFIG['cpu_threads']):
                for text in sample.text.iloc[:20]: predict_loaded(model, tok, [text], name, 'cpu')
                elapsed = []
                for _ in range(CONFIG['benchmark_repeats']):
                    for text in sample.text:
                        tick = time.perf_counter(); predict_loaded(model, tok, [text], name, 'cpu')
                        elapsed.append((time.perf_counter()-tick)*1000)
            results.append({'source': source, 'model': name, 'device': 'cpu', 'threads': CONFIG['cpu_threads'],
                'batch_size': 1, 'n_unique_texts': len(sample), 'repeats': CONFIG['benchmark_repeats'],
                'median_ms': float(np.median(elapsed)), 'p95_ms': float(np.percentile(elapsed,95)),
                'scope': 'cleaning + tokenizer/vectorizer + forward; no loading or download',
                'sample_ids': sample.sample_id.tolist(), 'hardware': hardware()})
            log(f'CPU benchmark {source}/{name}: median={np.median(elapsed):.2f}ms p95={np.percentile(elapsed,95):.2f}ms')
            del model, tok; gc.collect()
    write_json(ROOT / 'results' / 'benchmark.json', {**provenance(), 'created_utc': utcnow(), 'rows': results})

def environment():
    versions = {}
    for package in ['torch','transformers','datasets','scikit-learn','numpy','pandas','pyarrow','streamlit','matplotlib','huggingface-hub','safetensors','joblib','scipy','threadpoolctl','requests']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    write_json(ROOT / 'environment.json', {'python': platform.python_version(), 'packages': versions, 'hardware': hardware(), 'recorded_utc': utcnow()})
    (ROOT / 'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items() if v is not None)+'\n', 'utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','baselines','smoke','transformers','freeze','evaluate','benchmark','all'])
    args = parser.parse_args()
    (ROOT / 'results').mkdir(exist_ok=True)
    environment()
    stages = {'prepare': prepare, 'baselines': baselines, 'smoke': lambda: transformers_stage(True),
              'transformers': transformers_stage, 'freeze': freeze, 'evaluate': evaluate, 'benchmark': benchmark}
    if args.stage == 'all':
        for name in ['prepare','baselines','smoke','transformers','freeze','evaluate','benchmark']: stages[name]()
    else: stages[args.stage]()

if __name__ == '__main__': main()
