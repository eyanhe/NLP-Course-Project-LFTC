"""Train-only Word2Vec features and a packed bidirectional LSTM classifier."""
from collections import Counter
import re
import numpy as np
import torch
from torch import nn

def words(text): return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())

def mean_vectors(texts, vectors, dim=100):
    result=np.zeros((len(texts),dim),dtype=np.float32)
    for i,text in enumerate(texts):
        known=[vectors[w] for w in words(text) if w in vectors]
        if known: result[i]=np.mean(known,axis=0)
    return result

def word_rows(texts, vocabulary, max_length=256):
    return [torch.tensor(([vocabulary.get(w,1) for w in words(t)][:max_length] or [1]),dtype=torch.long) for t in texts]

def word_batch(items):
    sequences,labels=zip(*items)
    lengths=torch.tensor([len(x) for x in sequences],dtype=torch.long)
    return nn.utils.rnn.pad_sequence(sequences,batch_first=True,padding_value=0),lengths,torch.tensor(labels,dtype=torch.long)

class BiLSTM(nn.Module):
    def __init__(self, matrix, hidden_size=64, n_classes=2, dropout=.3):
        super().__init__()
        self.embedding=nn.Embedding.from_pretrained(torch.as_tensor(matrix,dtype=torch.float32),freeze=False,padding_idx=0)
        self.encoder=nn.LSTM(matrix.shape[1],hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden_size*2,n_classes)
    def forward(self, ids, lengths):
        embedding=self.embedding(ids)
        packed=nn.utils.rnn.pack_padded_sequence(embedding,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_)=self.encoder(packed)
        return self.classifier(self.dropout(torch.cat([hidden[-2],hidden[-1]],dim=1)))
