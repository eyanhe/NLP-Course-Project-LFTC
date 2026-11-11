"""Order-aware sentence-pair inputs for traditional features and a shared BiLSTM."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import nn
from models_extra import mean_vectors,word_rows

class PairTfidf(BaseEstimator,TransformerMixin):
    def __init__(self,ngram_range=(1,2),max_features=30000):
        self.ngram_range=ngram_range;self.max_features=max_features
    def fit(self,pairs,y=None):
        self.vectorizer_=TfidfVectorizer(ngram_range=self.ngram_range,max_features=self.max_features,min_df=2,sublinear_tf=True,dtype=np.float32)
        self.vectorizer_.fit([r['premise'] for r in pairs]+[r['hypothesis'] for r in pairs]);return self
    def transform(self,pairs):
        p=self.vectorizer_.transform([r['premise'] for r in pairs]);h=self.vectorizer_.transform([r['hypothesis'] for r in pairs])
        delta=p-h;delta.data=np.abs(delta.data)
        return sparse.hstack([p,h,delta,p.multiply(h)],format='csr')

class PairWordMeans(BaseEstimator,TransformerMixin):
    def __init__(self,vectors):self.vectors=vectors
    def fit(self,pairs,y=None):return self
    def transform(self,pairs):
        p=mean_vectors([r['premise'] for r in pairs],self.vectors);h=mean_vectors([r['hypothesis'] for r in pairs],self.vectors)
        return np.concatenate([p,h,np.abs(p-h),p*h],axis=1)

def pair_rows(pairs,vocabulary,max_length=256,labels=None):
    p=word_rows([r['premise'] for r in pairs],vocabulary,max_length);h=word_rows([r['hypothesis'] for r in pairs],vocabulary,max_length)
    return list(zip(p,h,labels if labels is not None else [0]*len(pairs)))

def pair_collate(items):
    p,h,y=zip(*items)
    return (nn.utils.rnn.pad_sequence(p,batch_first=True),torch.tensor([len(v) for v in p]),
            nn.utils.rnn.pad_sequence(h,batch_first=True),torch.tensor([len(v) for v in h]),torch.tensor(y,dtype=torch.long))

class PairBiLSTM(nn.Module):
    def __init__(self,matrix,hidden=64,dropout=.3):
        super().__init__()
        self.embedding=nn.Embedding.from_pretrained(torch.as_tensor(matrix,dtype=torch.float32),freeze=False,padding_idx=0)
        self.encoder=nn.LSTM(matrix.shape[1],hidden,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden*8,hidden*2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden*2,3))
    def encode(self,ids,lengths):
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(ids),lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_)=self.encoder(packed);return torch.cat([hidden[-2],hidden[-1]],dim=1)
    def forward(self,p,pl,h,hl):
        premise=self.encode(p,pl);hypothesis=self.encode(h,hl)
        return self.head(torch.cat([premise,hypothesis,torch.abs(premise-hypothesis),premise*hypothesis],dim=1))
