"""Inference compatibility for the preserved course experiment checkpoints.

No archived modules are imported. Torch is imported only when building a neural
model; traditional sentence-pair checkpoints need only NumPy/scikit-learn.
"""
import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())


def mean_vectors(texts, vectors):
    dimension = len(next(iter(vectors.values()))) if vectors else 100
    result = np.zeros((len(texts), dimension), dtype=np.float32)
    for i, text in enumerate(texts):
        known = [vectors[word] for word in words(text) if word in vectors]
        if known:
            result[i] = np.mean(known, axis=0)
    return result


class PairTfidf(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(1, 2), max_features=30000):
        self.ngram_range, self.max_features = ngram_range, max_features

    def fit(self, pairs, y=None):
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range, max_features=self.max_features,
            min_df=2, sublinear_tf=True, dtype=np.float32)
        self.vectorizer_.fit([r['premise'] for r in pairs] + [r['hypothesis'] for r in pairs])
        return self

    def transform(self, pairs):
        premise = self.vectorizer_.transform([r['premise'] for r in pairs])
        hypothesis = self.vectorizer_.transform([r['hypothesis'] for r in pairs])
        difference = premise - hypothesis
        difference.data = np.abs(difference.data)
        return sparse.hstack(
            [premise, hypothesis, difference, premise.multiply(hypothesis)], format='csr')


class PairWordMeans(BaseEstimator, TransformerMixin):
    def __init__(self, vectors):
        self.vectors = vectors

    def fit(self, pairs, y=None):
        return self

    def transform(self, pairs):
        premise = mean_vectors([r['premise'] for r in pairs], self.vectors)
        hypothesis = mean_vectors([r['hypothesis'] for r in pairs], self.vectors)
        return np.concatenate(
            [premise, hypothesis, np.abs(premise-hypothesis), premise*hypothesis], axis=1)


def bilstm_from_state(state, pair=False):
    """Reconstruct original layer names and dimensions directly from saved tensors."""
    import torch
    from torch import nn

    hidden_size = state['encoder.weight_hh_l0'].shape[1]

    class OriginalBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(
                state['embedding.weight'], freeze=False, padding_idx=0)
            self.encoder = nn.LSTM(self.embedding.embedding_dim, hidden_size,
                                   batch_first=True, bidirectional=True)
            if pair:
                self.head = nn.Sequential(
                    nn.Dropout(.3), nn.Linear(hidden_size*8, hidden_size*2),
                    nn.ReLU(), nn.Dropout(.3),
                    nn.Linear(hidden_size*2, state['head.4.weight'].shape[0]))
            else:
                self.dropout = nn.Dropout(.3)
                self.classifier = nn.Linear(hidden_size*2, state['classifier.weight'].shape[0])

        def encode(self, ids, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.encoder(packed)
            return torch.cat([hidden[-2], hidden[-1]], dim=1)

        def forward(self, ids, lengths, hypothesis=None, hypothesis_lengths=None):
            premise = self.encode(ids, lengths)
            if pair:
                other = self.encode(hypothesis, hypothesis_lengths)
                return self.head(torch.cat(
                    [premise, other, torch.abs(premise-other), premise*other], dim=1))
            return self.classifier(self.dropout(premise))

    model = OriginalBiLSTM()
    model.load_state_dict(state)
    return model.cpu().eval()
