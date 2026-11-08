"""Features learned exclusively from the supplied common-visible training rows."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .types import VisibleInput


def visible_texts(inputs) -> list[str]:
    values=[]
    for row in inputs:
        if not isinstance(row,VisibleInput):
            raise TypeError('Learners accept VisibleInput records, not raw Example text')
        values.append(row.text)
    return values


def words(text: str) -> list[str]:
    return re.findall(r'\w+|[^\w\s]',text.casefold(),flags=re.UNICODE)


def stable_word_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8],'little')


@dataclass
class TfidfFeatures:
    vectorizer: Any
    vocabulary: dict[str,int]
    degenerate: bool
    fitted_sample_ids: tuple[str,...]

    def transform(self,inputs):
        texts=visible_texts(inputs)
        return csr_matrix((len(texts),1),dtype=np.float32) if self.degenerate else self.vectorizer.transform(texts)


@dataclass
class WordFeatures:
    vocabulary: dict[str,int]
    embedding_matrix: np.ndarray
    degenerate: bool
    fitted_sample_ids: tuple[str,...]
    settings: dict[str,Any]

    def sequences(self,inputs) -> list[np.ndarray]:
        # The character-visible input was already fixed. Do not truncate its
        # independently tokenized word sequence a second time.
        return [np.asarray([self.vocabulary.get(word,1) for word in words(text)] or [1],dtype=np.int64)
                for text in visible_texts(inputs)]

    def transform(self,inputs) -> np.ndarray:
        texts=visible_texts(inputs)
        result=np.zeros((len(texts),self.embedding_matrix.shape[1]),dtype=np.float32)
        for i,text in enumerate(texts):
            ids=[self.vocabulary[word] for word in words(text) if word in self.vocabulary]
            if ids:
                result[i]=self.embedding_matrix[ids].mean(axis=0)
        return result


def fit_features(model_id: str,train_inputs: tuple[VisibleInput,...],hp: dict[str,Any]):
    inputs=tuple(sorted(train_inputs,key=lambda row:row.sample_id))
    texts=visible_texts(inputs)
    ids=tuple(row.sample_id for row in inputs)
    if model_id in {'tfidf_lr','tfidf_svm','fixed_tfidf_svm'}:
        vectorizer=TfidfVectorizer(max_features=int(hp.get('max_features',30000)),
            min_df=int(hp.get('min_df',1)),sublinear_tf=bool(hp.get('sublinear_tf',True)),
            ngram_range=tuple(hp.get('ngram_range',(1,2))),dtype=np.float32)
        try:
            vectorizer.fit(texts)
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc) and 'no terms remain' not in str(exc).lower():
                raise
            return TfidfFeatures(None,{},True,ids)
        return TfidfFeatures(vectorizer,dict(vectorizer.vocabulary_),False,ids)
    if model_id not in {'w2v_svm','bilstm'}:
        raise ValueError(f'This model does not learn a local feature space: {model_id}')
    settings={**hp.get('embedding_initialization',{}),**hp}
    dimension=int(settings.get('vector_size',100))
    sentences=[words(text) for text in texts]
    counts=Counter(token for sentence in sentences for token in sentence)
    allowed=sorted((word for word,n in counts.items() if n>=int(settings.get('min_count',1))),
                   key=lambda word:(-counts[word],word))[:int(settings.get('vocabulary_limit',30000))]
    vocabulary={word:i+2 for i,word in enumerate(allowed)}
    matrix=np.zeros((len(vocabulary)+2,dimension),dtype=np.float32)
    if vocabulary:
        from gensim.models import Word2Vec
        selected=set(vocabulary)
        corpus=[[word for word in sentence if word in selected] for sentence in sentences]
        corpus=[sentence for sentence in corpus if sentence]
        model=Word2Vec(sentences=corpus,vector_size=dimension,
            window=int(settings.get('window',5)),min_count=1,epochs=int(settings.get('epochs',10)),
            workers=1,sg=1,seed=int(settings.get('seed',2026)),hashfxn=stable_word_hash,sorted_vocab=1)
        for word,index in vocabulary.items():
            matrix[index]=model.wv[word]
    return WordFeatures(vocabulary,matrix,not bool(vocabulary),ids,
        {'vector_size':dimension,'workers':1,'sg':1,'hash_function':'sha256_first8_little_endian',
         'vocabulary_limit':int(settings.get('vocabulary_limit',30000)),
         'seed':int(settings.get('seed',2026)),'epochs':int(settings.get('epochs',10))})
