"""Shared encoders for single texts and ordered text pairs; imported lazily."""
import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, matrix, label_count, hidden_size=64, dropout=.3, pair=False):
        super().__init__()
        self.pair = pair
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(matrix, dtype=torch.float32),
                                                       freeze=False, padding_idx=0)
        self.encoder = nn.LSTM(matrix.shape[1], hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size*(8 if pair else 2), label_count))

    def encode(self, ids, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(ids), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

    def forward(self, ids, lengths, pair_ids=None, pair_lengths=None):
        left = self.encode(ids, lengths)
        if self.pair:
            right = self.encode(pair_ids, pair_lengths)
            left = torch.cat((left, right, torch.abs(left-right), left*right), dim=1)
        return self.head(left)
