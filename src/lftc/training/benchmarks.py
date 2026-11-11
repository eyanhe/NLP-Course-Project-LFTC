"""Local inference for the original five-model, five-dataset PPT benchmark.

Loading never trains, downloads, imports archive scripts, or executes historical
protocol checks. Original checkpoints remain in their preserved directories.
"""
import html
import json
from pathlib import Path
import re
from time import perf_counter


MODEL_IDS = ('logistic_regression', 'linear_svm', 'word2vec_svm', 'bilstm', 'distilbert')
SOURCES = ('imdb', 'rotten_tomatoes', 'yelp', 'ag_news', 'mnli')
LABELS = {source: ('Negative', 'Positive') for source in SOURCES[:3]}
LABELS.update(ag_news=('World', 'Sports', 'Business', 'Sci/Tech'),
              mnli=('Entailment', 'Neutral', 'Contradiction'))
MAX_LENGTH = 256


def clean_text(text):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', str(text)))).strip()


def _validate(source, model_id):
    if source not in SOURCES:
        raise ValueError(f'Unknown benchmark source: {source}')
    if model_id not in MODEL_IDS:
        raise ValueError(f'Unknown benchmark model: {model_id}')


def _roots(root):
    project = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    preserved = project / 'archive' / '2026-11-08'
    return project, preserved if preserved.is_dir() else project


def _directory(base, source, model_id):
    name = f'{model_id}_seed42' if model_id in MODEL_IDS[3:] else model_id
    if source == 'mnli':
        return base / 'models-hard' / name
    area = ('models' if source != 'ag_news' and model_id in
            ('logistic_regression', 'linear_svm', 'distilbert') else 'models-v2')
    return base / area / source / name


def _load_joblib(path):
    """Map only the two historical feature classes, without global module aliases."""
    import joblib.numpy_pickle as numpy_pickle
    from lftc.models.benchmark_compat import PairTfidf, PairWordMeans

    class BenchmarkUnpickler(numpy_pickle.NumpyUnpickler):
        def find_class(self, module, name):
            if module == 'pair_models' and name in ('PairTfidf', 'PairWordMeans'):
                return {'PairTfidf': PairTfidf, 'PairWordMeans': PairWordMeans}[name]
            return super().find_class(module, name)

    with path.open('rb') as stream:
        if hasattr(numpy_pickle, '_read_fileobject'):  # pinned joblib 1.2
            with numpy_pickle._read_fileobject(stream, str(path), None) as readable:
                return BenchmarkUnpickler(str(path), readable, mmap_mode=None).load()
        # Newer joblib versions expose the same decompression through this helper.
        with numpy_pickle._validate_fileobject_and_memmap(stream, str(path), None) as result:
            readable, _ = result
            return BenchmarkUnpickler(str(path), readable, ensure_native_byte_order=True,
                                      mmap_mode=None).load()


class BenchmarkModel:
    def __init__(self, source, model_id, model, labels, tokenizer=None, vocabulary=None):
        self.source, self.model_id, self.model = source, model_id, model
        self.labels, self.tokenizer, self.vocabulary = tuple(labels), tokenizer, vocabulary

    def predict(self, text, premise=''):
        import numpy as np
        from lftc.models.benchmark_compat import mean_vectors, words

        text, premise = clean_text(text), clean_text(premise)
        pair = self.source == 'mnli'
        if not text:
            raise ValueError('Enter nonempty text for benchmark inference')
        if pair and not premise:
            raise ValueError('MultiNLI inference requires both premise and hypothesis')
        truncated = False
        if self.model_id == 'bilstm':
            truncated = max(len(words(text)), len(words(premise)) if pair else 0) > MAX_LENGTH
        elif self.model_id == 'distilbert':
            uncut = self.tokenizer(premise, text, truncation=False, verbose=False) if pair else self.tokenizer(
                text, truncation=False, verbose=False)
            truncated = len(uncut['input_ids']) > MAX_LENGTH

        start = perf_counter()
        if self.model_id in MODEL_IDS[3:]:
            import torch
            with torch.inference_mode():
                if self.model_id == 'bilstm':
                    def encoded(value):
                        ids = [self.vocabulary.get(word, 1) for word in words(value)][:MAX_LENGTH] or [1]
                        return torch.tensor([ids], dtype=torch.long), torch.tensor([len(ids)])

                    logits = self.model(*encoded(premise), *encoded(text)) if pair else self.model(*encoded(text))
                else:
                    args = (premise, text) if pair else (text,)
                    batch = self.tokenizer(*args, truncation='longest_first' if pair else True,
                                           max_length=MAX_LENGTH, padding=True, return_tensors='pt')
                    logits = self.model(**batch).logits
                values = logits.float().softmax(-1).cpu().numpy()[0]
            label = int(values.argmax())
            score, score_label = float(values[label]), 'predicted-class probability'
        else:
            model = self.model
            inputs = [{'premise': premise, 'hypothesis': text}] if pair else [text]
            if self.model_id == 'word2vec_svm' and not pair:
                inputs, model = mean_vectors([text], model['vectors']), model['classifier']
            label = int(model.predict(inputs)[0])
            probability = self.model_id == 'logistic_regression'
            values = np.asarray(model.predict_proba(inputs) if probability else model.decision_function(inputs))
            index = list(model.classes_).index(label)
            if probability:
                score, score_label = float(values[0, index]), 'predicted-class probability'
            elif values.ndim == 1:
                score, score_label = float(values[0]), 'positive-class margin'
            else:
                score, score_label = float(values[0, index]), 'predicted-class margin'
        elapsed = (perf_counter() - start) * 1000
        return {'prediction': self.labels[label], 'label': label, 'score': score,
                'score_label': score_label, 'milliseconds': elapsed, 'truncated': truncated}


def load_benchmark_model(source, model_id, root=None):
    """Load the preserved selected model (neural seed 42) for CPU inference."""
    _validate(source, model_id)
    project, base = _roots(root)
    directory = _directory(base, source, model_id)
    required = ('model.pt', 'tokens.joblib') if model_id == 'bilstm' else (
        ('config.json',) if model_id == 'distilbert' else ('model.joblib',))
    for filename in required:
        if not (directory / filename).is_file():
            raise FileNotFoundError(
                f'Missing original benchmark checkpoint for {source}/{model_id}: {directory / filename}')
    metadata = json.loads((directory / 'metadata.json').read_text('utf-8')) if (
        directory / 'metadata.json').is_file() else {}
    labels = metadata.get('labels', LABELS[source])
    if source == 'mnli':
        config = base / 'legacy-root' / 'config-hard.json'
        if config.is_file():
            labels = json.loads(config.read_text('utf-8')).get('labels', labels)
    if model_id == 'bilstm':
        import torch
        from lftc.models.benchmark_compat import bilstm_from_state
        torch.set_num_threads(4)
        tokens = _load_joblib(directory / 'tokens.joblib')
        state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
        model = bilstm_from_state(state, pair=source == 'mnli')
        return BenchmarkModel(source, model_id, model, labels, vocabulary=tokens['vocabulary'])
    if model_id == 'distilbert':
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(4)
        if not any((directory / name).is_file() for name in
                   ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json',
                    'pytorch_model.bin.index.json')):
            raise FileNotFoundError(f'Missing original benchmark weights for {source}/{model_id}: {directory}')
        tokenizer_dir = directory
        if not any((directory / name).is_file() for name in ('tokenizer.json', 'vocab.txt')):
            tokenizer_dir = project / 'artifacts' / 'assets' / 'models' / 'distilbert'
            if not tokenizer_dir.is_dir():
                raise FileNotFoundError(f'Missing local benchmark tokenizer: {tokenizer_dir}')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, attn_implementation='eager').cpu().eval()
        saved_labels = model.config.id2label
        if saved_labels and not all(str(value).startswith('LABEL_') for value in saved_labels.values()):
            labels = [saved_labels[i] for i in range(model.config.num_labels)]
            if source in SOURCES[:3]:
                labels = [str(value).capitalize() for value in labels]
        return BenchmarkModel(source, model_id, model, labels, tokenizer=tokenizer)
    return BenchmarkModel(source, model_id, _load_joblib(directory / 'model.joblib'), labels)


def predict_benchmarks(source, text, premise='', models=None, root=None, loader=None):
    """Compare existing models; an injected loader may cache models across UI reruns."""
    names = MODEL_IDS if models is None else tuple(models)
    for name in names:
        _validate(source, name)
    if source not in SOURCES:
        raise ValueError(f'Unknown benchmark source: {source}')
    rows = []
    for name in names:
        model = loader(source, name) if loader is not None else load_benchmark_model(source, name, root)
        result = model.predict(text, premise=premise)
        rows.append({'model': name, **{key: result[key] for key in
                     ('prediction', 'score', 'score_label', 'milliseconds', 'truncated')}})
    return rows
