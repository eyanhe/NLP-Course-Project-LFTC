"""Train-only lexical, Word2Vec and sequence features."""
from collections import Counter
from dataclasses import dataclass
import re
import zlib

import numpy as np


def words(text):
    return re.findall(r'\w+|[^\w\s]', text.casefold(), flags=re.UNICODE)


def _word_hash(text):
    return zlib.crc32(text.encode('utf-8'))


def _pair_join(left, right):
    from scipy import sparse
    if sparse.issparse(left):
        delta = left-right
        delta.data = np.abs(delta.data)
        return sparse.hstack((left, right, delta, left.multiply(right)), format='csr')
    return np.concatenate((left, right, np.abs(left-right), left*right), axis=1)


@dataclass
class TfidfFeatures:
    vectorizer: object
    vocabulary: dict
    degenerate: bool
    pair: bool
    fitted_sample_ids: tuple

    def transform(self, examples):
        from scipy.sparse import csr_matrix
        rows = tuple(examples)
        if self.degenerate or not rows:
            width = max(1, len(self.vocabulary)) * (4 if self.pair else 1)
            return csr_matrix((len(rows), width), dtype=np.float32)
        left = self.vectorizer.transform([row.text for row in rows])
        return _pair_join(left, self.vectorizer.transform([row.text_pair or '' for row in rows])) if self.pair else left


@dataclass
class WordFeatures:
    vocabulary: dict
    embedding_matrix: np.ndarray
    degenerate: bool
    pair: bool
    fitted_sample_ids: tuple

    def sequences(self, examples, max_length=256, *, second=False):
        return [np.asarray([self.vocabulary.get(word, 1) for word in
                           words((row.text_pair or '') if second else row.text)][:max_length] or [1], dtype=np.int64)
                for row in examples]

    def _means(self, texts):
        result = np.zeros((len(texts), self.embedding_matrix.shape[1]), dtype=np.float32)
        for i, text in enumerate(texts):
            ids = [self.vocabulary[word] for word in words(text) if word in self.vocabulary]
            if ids:
                result[i] = self.embedding_matrix[ids].mean(axis=0)
        return result

    def transform(self, examples):
        rows = tuple(examples)
        left = self._means([row.text for row in rows])
        return _pair_join(left, self._means([row.text_pair or '' for row in rows])) if self.pair else left


def fit_features(model_id, examples, options=None):
    options = dict(options or {})
    rows = tuple(examples)
    pair = any(row.text_pair is not None for row in rows)
    texts = [row.text for row in rows] + ([row.text_pair or '' for row in rows] if pair else [])
    ids = tuple(row.sample_id for row in rows)
    if model_id in ('tfidf_lr', 'tfidf_svm'):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=int(options.get('max_features', 30000)),
                                     min_df=options.get('min_df', 1),
                                     ngram_range=tuple(options.get('ngram_range', (1, 2))),
                                     sublinear_tf=True, dtype=np.float32)
        try:
            vectorizer.fit(texts)
        except ValueError as exc:
            if not any(value in str(exc).lower() for value in ('empty vocabulary', 'no terms remain', 'max_df corresponds')):
                raise
            return TfidfFeatures(None, {}, True, pair, ids)
        return TfidfFeatures(vectorizer, dict(vectorizer.vocabulary_), False, pair, ids)
    if model_id not in ('w2v_svm', 'bilstm'):
        raise ValueError(f'Unknown feature family: {model_id}')
    settings = {**options.get('embedding_initialization', {}), **options}
    dim = int(settings.get('vector_size', settings.get('embedding_dim', 100)))
    sentences = [words(text) for text in texts]
    counts = Counter(word for sentence in sentences for word in sentence)
    allowed = sorted((word for word, count in counts.items() if count >= int(settings.get('min_count', 1))),
                     key=lambda word: (-counts[word], word))[:int(settings.get('vocabulary_limit', 30000))]
    vocabulary = {word: i+2 for i, word in enumerate(allowed)}
    matrix = np.zeros((len(vocabulary)+2, dim), dtype=np.float32)
    if vocabulary:
        from gensim.models import Word2Vec
        corpus = [[word for word in sentence if word in vocabulary] for sentence in sentences]
        corpus = [sentence for sentence in corpus if sentence]
        model = Word2Vec(sentences=corpus, vector_size=dim, min_count=1, workers=1, sg=1,
                         window=int(settings.get('window', 5)),
                         epochs=int(settings.get('w2v_epochs', settings.get('embedding_epochs', 10))),
                         seed=int(settings.get('seed', 42)), hashfxn=_word_hash)
        for word, index in vocabulary.items():
            matrix[index] = model.wv[word]
    return WordFeatures(vocabulary, matrix, not bool(vocabulary), pair, ids)
