"""Real local model loading and synthetic inference only, with no optimizer."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, hash_file, read_json


def synthetic_visible_texts(tokenizer: Any,batch_size: int) -> list[str]:
    from .inputs import make_visible_input
    from .types import Example
    raw=['A synthetic check: I did not request this payment. 🙂 '*40,
         'A synthetic check about emotion, topic and intent.']
    return [make_visible_input(Example('synthetic',f'synthetic/{i}',raw[i%2],(0,),'synthetic'),tokenizer).text
            for i in range(batch_size)]


def check_forward(model: Any,batch: dict[str,Any],labels: int,task: str) -> dict[str,Any]:
    import torch
    before=tuple(p._version for p in model.parameters())
    model.eval()
    with torch.inference_mode():
        logits=model(**batch).logits
        shape=[int(n) for n in logits.shape]
        expected=[int(batch['input_ids'].shape[0]),labels]
        if shape!=expected:
            raise ValueError(f'Incorrect label shape: {shape}, expected {expected}')
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Non-finite synthetic forward outputs')
        if task=='multi_label':
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),torch.zeros_like(logits).float())
        else:
            loss=torch.nn.functional.cross_entropy(logits.float(),torch.zeros(shape[0],dtype=torch.long,device=logits.device))
    changed=tuple(p._version for p in model.parameters())!=before
    gradients=any(p.grad is not None for p in model.parameters())
    if changed or gradients:
        raise ValueError('Forward-only check changed model weights or created gradients')
    return {'logits_shape':shape,'finite_outputs':True,'finite_synthetic_loss':bool(torch.isfinite(loss)),
            'weight_versions_unchanged':not changed,'gradients_created':gradients,
            'actual_model_training_run':False,'synthetic_logits_only':True}


def run_forward_checks(p: dict[str,Any],root: Path) -> dict[str,Any]:
    import torch
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    assets=read_json(root/'assets/assets.lock.json')
    torch.set_num_threads(p['runtime']['cpu_threads'])
    if p['runtime']['device']=='cuda' and not torch.cuda.is_available():
        raise ValueError('Configured CUDA device is unavailable')
    device=p['runtime']['device']
    bf16=device=='cuda' and torch.cuda.is_bf16_supported()
    dtype=torch.bfloat16 if bf16 else torch.float32
    evidence={'stage':'G0a_forward_only','asset_hash':assets['asset_hash'],
              'protocol_hash':p['_protocol_hash'],'device':device,'dtype':str(dtype),
              'checks':[],'actual_model_training_run':False,
              'official_test_predictions_generated':False,
              'full_E0_training_smoke':'not_run_pretraining_scope'}
    common_tokenizer=AutoTokenizer.from_pretrained(assets['models']['distilbert']['directory'],local_files_only=True)
    texts=synthetic_visible_texts(common_tokenizer,p['runtime']['micro_batch'])
    for family in ['distilbert','modernbert']:
        directory=assets['models'][family]['directory']
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True)
        batch=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        batch={k:v.to(device) for k,v in batch.items() if k in ['input_ids','attention_mask']}
        for labels,task in [(77,'single_label'),(28,'multi_label')]:
            torch.manual_seed(42)
            config=AutoConfig.from_pretrained(directory,local_files_only=True)
            config.num_labels=labels
            config.problem_type='multi_label_classification' if task=='multi_label' else 'single_label_classification'
            if family=='modernbert':
                config.reference_compile=False
            model=AutoModelForSequenceClassification.from_pretrained(directory,config=config,
                   local_files_only=True,attn_implementation='sdpa',torch_dtype=dtype).to(device)
            if int(batch['input_ids'].shape[1])>config.max_position_embeddings:
                raise ValueError('Common visible text exceeds the reference model positional capacity')
            if device=='cuda':
                torch.cuda.reset_peak_memory_stats()
            result=check_forward(model,batch,labels,task)
            evidence['checks'].append({'family':family,'task':task,**result,
                'input_shape':list(batch['input_ids'].shape),'common_visible_text_preserved':True,
                'parameter_count':sum(n.numel() for n in model.parameters()),
                'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                'reference_compile':False if family=='modernbert' else None,
                'attention_implementation':model.config._attn_implementation})
            del model
            gc.collect()
            if device=='cuda':
                torch.cuda.empty_cache()
    from models_extra import BiLSTM
    import numpy as np
    for labels,task in [(77,'single_label'),(28,'multi_label')]:
        torch.manual_seed(42)
        model=BiLSTM(np.zeros((10,100),dtype=np.float32),n_classes=labels).to(device).eval()
        before=tuple(n._version for n in model.parameters())
        with torch.inference_mode():
            out=model(torch.tensor([[1,2,3],[1,2,0]],device=device),torch.tensor([3,2]))
        passed=list(out.shape)==[2,labels] and bool(torch.isfinite(out).all())
        passed=passed and before==tuple(n._version for n in model.parameters()) and all(n.grad is None for n in model.parameters())
        if not passed:
            raise ValueError('BiLSTM forward contract failed')
        evidence['checks'].append({'family':'bilstm','task':task,'logits_shape':list(out.shape),
                                  'finite_outputs':True,'gradients_created':False,'weight_versions_unchanged':True})
        del model
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from gensim.models import Word2Vec
    LogisticRegression(C=1,max_iter=1000)
    LinearSVC(C=1,max_iter=10000)
    Word2Vec(vector_size=100,window=5,min_count=1,workers=1,sg=1,seed=2026)
    evidence['traditional_estimators_constructed_without_fit']=['tfidf_lr','tfidf_svm','w2v_svm']
    evidence['all_passed']=all(c['finite_outputs'] and c['weight_versions_unchanged'] and not c['gradients_created']
                              for c in evidence['checks'])
    atomic_write_json(root/'forward_checks.json',evidence,root)
    return evidence
