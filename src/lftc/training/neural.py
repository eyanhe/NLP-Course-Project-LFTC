"""Small batched neural training with validation loss early stopping.

Transformer sources and reloads are strictly local. Checkpoints contain native
package classes and their own tokenizer/model files, never archive imports.
"""
import json
from pathlib import Path
import random

import numpy as np
import torch

from lftc.domain import TaskType
from lftc.features import fit_features
from lftc.models.neural import BiLSTM
from . import Predictor, target_array


class NeuralPredictor(Predictor):
    def __init__(self, model_id, task, label_count, model, options, features=None, tokenizer=None, constants=None):
        self.model_id, self.task, self.label_count = model_id, TaskType(task), label_count
        self.model, self.features, self.tokenizer = model, features, tokenizer
        self.options, self.constants = dict(options), dict(constants or {})
        self.score_kind = 'logit'
        self.history = []
        self.selected_epoch = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _batch(self, rows):
        maximum = int(self.options.get('max_length', 256))
        if self.model_id == 'bilstm':
            def encoded(second=False):
                sequences = [torch.as_tensor(seq) for seq in self.features.sequences(rows, maximum, second=second)]
                ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(self.device)
                lengths = torch.as_tensor([len(seq) for seq in sequences], dtype=torch.long)
                return ids, lengths
            ids, lengths = encoded()
            batch = {'ids': ids, 'lengths': lengths}
            if self.features.pair:
                batch['pair_ids'], batch['pair_lengths'] = encoded(True)
            return batch
        pairs = [row.text_pair or '' for row in rows] if self.task == TaskType.PAIR_CLASSIFICATION else None
        batch = self.tokenizer([row.text for row in rows], text_pair=pairs,
                               truncation=True, padding=True, max_length=maximum, return_tensors='pt')
        return {key: value.to(self.device) for key, value in batch.items()}

    def _logits(self, rows):
        output = self.model(**self._batch(rows))
        return output if self.model_id == 'bilstm' else output.logits

    def _constant_scores(self, scores):
        for label, value in self.constants.items():
            scores[:, int(label)] = float(value)
        return scores

    def predict_scores(self, examples, batch_size=32):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        rows = tuple(examples)
        if not rows:
            return np.empty((0, self.label_count))
        self.model.eval()
        with torch.inference_mode():
            scores = np.concatenate([self._logits(rows[i:i+batch_size]).float().cpu().numpy()
                                     for i in range(0, len(rows), batch_size)])
        return self._constant_scores(scores)

    def save(self, directory):
        import joblib
        directory = Path(directory)
        metadata = {'options': self.options, 'constants': self.constants,
                    'history': self.history, 'selected_epoch': self.selected_epoch}
        (directory/'neural.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        if self.model_id == 'bilstm':
            joblib.dump(self.features, directory/'features.joblib')
            torch.save({key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                       directory/'weights.pt')
        else:
            self.model.save_pretrained(directory/'transformer')
            self.tokenizer.save_pretrained(directory/'transformer')


def _make_bilstm(features, label_count, options):
    return BiLSTM(features.embedding_matrix.copy(), label_count,
                  hidden_size=int(options.get('hidden_size', options.get('hidden_per_direction', 64))),
                  dropout=float(options.get('dropout', .3)), pair=features.pair)


def _local_source(model_dir, options):
    value = model_dir or options.get('local_model_dir') or options.get('model_path') or options.get('source')
    if value is None or not Path(value).is_dir():
        raise ValueError('Transformer training requires an existing local model directory')
    return Path(value)


def train_neural(train, validation, task, label_count, model_id, options, model_dir):
    seed = int(options.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(options.get('cpu_threads', 4)))
    maximum = int(options.get('max_length', 256))
    epochs = int(options.get('epochs', options.get('max_epochs', 10 if model_id == 'bilstm' else 3)))
    batch_size = int(options.get('batch_size', 32 if model_id == 'bilstm' else 8))
    patience = int(options.get('patience', 2))
    if min(maximum, epochs, batch_size, patience) < 1:
        raise ValueError('max_length, epochs, batch_size and patience must be positive')
    features = tokenizer = None
    if model_id == 'bilstm':
        features = fit_features(model_id, train, options)
        model = _make_bilstm(features, label_count, options)
    else:
        source = _local_source(model_dir, options)
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        config = AutoConfig.from_pretrained(source, local_files_only=True)
        config.num_labels = label_count
        config.problem_type = 'multi_label_classification' if task == TaskType.MULTILABEL else 'single_label_classification'
        if model_id == 'modernbert':
            config.reference_compile = False
        model = AutoModelForSequenceClassification.from_pretrained(
            source, config=config, local_files_only=True, ignore_mismatched_sizes=True,
            attn_implementation=options.get('attn_implementation', 'sdpa'))
    device = str(options.get('device', 'cpu'))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    model.to(device)
    y = target_array(train, task, label_count)
    vy = target_array(validation, task, label_count) if validation else None
    if task == TaskType.MULTILABEL:
        constants = {i: (30. if y[0, i] else -30.) for i in range(label_count) if np.all(y[:, i] == y[0, i])}
    else:
        constants = {i: -1e9 for i in range(label_count) if i not in set(y.tolist())}
    predictor = NeuralPredictor(model_id, task, label_count, model, options, features, tokenizer, constants)
    active = [i for i in range(label_count) if i not in constants]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options.get('learning_rate', options.get('lr',
                                  .001 if model_id == 'bilstm' else 2e-5))),
                                  weight_decay=float(options.get('weight_decay', 0 if model_id == 'bilstm' else .01)))

    def loss_for(logits, targets):
        if task == TaskType.MULTILABEL:
            return (torch.nn.functional.binary_cross_entropy_with_logits(logits[:, active], targets[:, active].float())
                    if active else logits.sum()*0)
        return torch.nn.functional.cross_entropy(logits, targets.long())

    best, stale, best_state = float('inf'), 0, None
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.
        order = rng.permutation(len(train))
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset+batch_size]
            rows = tuple(train[int(i)] for i in indices)
            targets = torch.as_tensor(y[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(predictor._logits(rows), targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite neural training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(options.get('gradient_clip', 1.)))
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        validation_loss = None
        if validation:
            model.eval()
            validation_loss = 0.
            with torch.inference_mode():
                for offset in range(0, len(validation), batch_size):
                    rows = validation[offset:offset+batch_size]
                    targets = torch.as_tensor(vy[offset:offset+batch_size], device=device)
                    validation_loss += float(loss_for(predictor._logits(rows), targets))*len(rows)
            validation_loss /= len(validation)
        predictor.history.append({'epoch': epoch, 'train_loss': total_loss/len(train), 'validation_loss': validation_loss})
        value = validation_loss if validation else total_loss/len(train)
        if value < best - float(options.get('min_delta', 0)):
            best, stale = value, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            predictor.selected_epoch = epoch
        else:
            stale += 1
        if validation and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return predictor


def load_neural(directory, metadata):
    import joblib
    record = json.loads((directory/'neural.json').read_text(encoding='utf-8'))
    options = {**record['options'], 'device': 'cpu'}
    features = tokenizer = None
    if metadata['model_id'] == 'bilstm':
        features = joblib.load(directory/'features.joblib')
        model = _make_bilstm(features, metadata['label_count'], options)
        model.load_state_dict(torch.load(directory/'weights.pt', map_location='cpu', weights_only=True))
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        source = directory/'transformer'
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True,
                    attn_implementation=options.get('attn_implementation', 'sdpa'))
    predictor = NeuralPredictor(metadata['model_id'], metadata['task'], metadata['label_count'], model,
                                options, features, tokenizer, record['constants'])
    predictor.history = record['history']
    predictor.selected_epoch = record['selected_epoch']
    predictor.model.eval()
    return predictor
