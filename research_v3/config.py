"""Protocol validation and explicit gates before any fitting or test prediction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_DATASETS = {'go_emotions', 'banking77', 'yahoo_answers_topics'}
MAIN_MODELS = {'tfidf_lr', 'tfidf_svm', 'w2v_svm', 'bilstm', 'distilbert'}


def canonical_hash(value: Any) -> str:
    public = {k: v for k, v in value.items() if not k.startswith('_')} if isinstance(value, dict) else value
    encoded = json.dumps(public, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def output_root(p: dict[str, Any]) -> Path:
    return (PROJECT_ROOT / p.get('output_root', 'research-v3')).resolve()


def validate_protocol(p: dict[str, Any]) -> None:
    scope = p.get('scope', {})
    new = scope.get('new_datasets', [])
    if len(new) != 3 or set(new) != NEW_DATASETS:
        raise ValueError('Exactly the three registered new datasets are required')
    models = scope.get('main_models', [])
    if len(models) != 5 or set(models) != MAIN_MODELS:
        raise ValueError('The five main model families must remain explicit')
    if output_root(p) != (PROJECT_ROOT / 'research-v3').resolve():
        raise ValueError('output_root must be the isolated project research-v3 directory')
    for dataset in NEW_DATASETS:
        source = p.get('sources', {}).get(dataset, {})
        if not re.fullmatch('[a-f0-9]{40}', source.get('revision', '')):
            raise ValueError(f'{dataset}: an immutable source revision is required')
    if p.get('partitions', {}).get('strict_fewshot_target_validation_access') is not False:
        raise ValueError('Strict few-shot validation access must be disabled')
    if p.get('input', {}).get('max_length_with_special_tokens') != 256:
        raise ValueError('The fixed main input budget must be 256 including special tokens')
    if p.get('sources', {}).get('go_emotions', {}).get('label_count') != 28:
        raise ValueError('GoEmotions must preserve its 28-label target space')
    if p.get('freezes', {}).get('scope') != 'global_all_required_experiments':
        raise ValueError('The global three-stage freeze contract is required')
    execution = p.get('execution', {})
    if execution.get('stage', 'pretraining') == 'pretraining' and execution.get('training_enabled', False):
        raise ValueError('A pretraining configuration cannot enable fitting')


def load_protocol(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    p = json.loads(path.read_text(encoding='utf-8-sig'))
    p.setdefault('output_root', 'research-v3')
    p.setdefault('execution', {'stage': 'pretraining', 'training_enabled': False,
                               'test_predictions_enabled': False})
    validate_protocol(p)
    p['_config_path'] = str(path)
    p['_project_root'] = str(PROJECT_ROOT)
    p['_protocol_hash'] = canonical_hash(p)
    return p


def require_training_enabled(p: dict[str, Any]) -> None:
    if not p.get('execution', {}).get('training_enabled', False):
        raise PermissionError('Actual model fitting is disabled: this is the pretraining preparation stage')


def require_test_enabled(p: dict[str, Any]) -> None:
    if not p.get('execution', {}).get('test_predictions_enabled', False):
        raise PermissionError('Official-test model predictions are disabled before evaluation.freeze')
    if not (output_root(p) / 'evaluation.freeze.json').is_file():
        raise PermissionError('A verified evaluation.freeze.json is required')
