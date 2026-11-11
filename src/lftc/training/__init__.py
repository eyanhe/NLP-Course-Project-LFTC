"""Training and portable checkpoints; optional neural imports remain lazy."""
import json
from pathlib import Path

import numpy as np

from lftc.domain import TaskType


class Predictor:
    def predict(self, examples, threshold=None):
        scores = self.predict_scores(examples)
        if self.task == TaskType.MULTILABEL:
            if threshold is None:
                threshold = .5 if self.score_kind == 'probability' else 0.
            return (scores >= threshold).astype(int)
        return scores.argmax(axis=1)


class ClassicalPredictor(Predictor):
    def __init__(self, model_id, task, label_count, features, model):
        self.model_id, self.task, self.label_count = model_id, TaskType(task), label_count
        self.features, self.model, self.score_kind = features, model, model.score_kind

    def predict_scores(self, examples, batch_size=32):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        rows = tuple(examples)
        if not rows:
            return np.empty((0, self.label_count))
        return np.concatenate([self.model.scores(self.features.transform(rows[i:i+batch_size]))
                               for i in range(0, len(rows), batch_size)])


def target_array(examples, task, label_count):
    rows = tuple(examples)
    if any(any(label < 0 or label >= label_count for label in row.labels) for row in rows):
        raise ValueError('Training labels outside declared label space')
    if task == TaskType.MULTILABEL:
        result = np.zeros((len(rows), label_count), dtype=np.float32)
        for i, row in enumerate(rows):
            result[i, list(row.labels)] = 1
        return result
    if any(len(row.labels) != 1 for row in rows):
        raise ValueError('Single-label tasks require exactly one label per row')
    return np.asarray([row.labels[0] for row in rows], dtype=np.int64)


def train_model(train, validation, task, label_count, model_id, options=None, model_dir=None):
    from lftc.features import fit_features
    from lftc.models import TraditionalClassifier
    train, validation, task = tuple(train), tuple(validation), TaskType(task)
    options = dict(options or {})
    if not train or label_count < 2:
        raise ValueError('Nonempty training rows and at least two labels required')
    y = target_array(train, task, label_count)
    if model_id in ('tfidf_lr', 'tfidf_svm', 'w2v_svm'):
        features = fit_features(model_id, train, options)
        model = TraditionalClassifier(model_id, task, label_count, options).fit(features.transform(train), y)
        return ClassicalPredictor(model_id, task, label_count, features, model)
    if model_id not in ('bilstm', 'distilbert', 'modernbert'):
        raise ValueError(f'Unknown model family: {model_id}')
    from .neural import train_neural
    return train_neural(train, validation, task, label_count, model_id, options, model_dir)


def save_model(predictor, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(predictor, ClassicalPredictor):
        import joblib
        joblib.dump(predictor, directory/'model.joblib')
        backend = 'classical'
    else:
        predictor.save(directory)
        backend = 'neural'
    (directory/'model.json').write_text(json.dumps({
        'schema_version': 1, 'backend': backend, 'model_id': predictor.model_id,
        'task': predictor.task.value, 'label_count': predictor.label_count,
        'score_kind': predictor.score_kind}, indent=2), encoding='utf-8')


def load_model(directory):
    """Load only checkpoints from a trusted source (joblib is executable)."""
    directory = Path(directory)
    metadata = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported model schema')
    if metadata['backend'] == 'classical':
        import joblib
        return joblib.load(directory/'model.joblib')
    from .neural import load_neural
    return load_neural(directory, metadata)
