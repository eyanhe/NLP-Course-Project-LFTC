"""Auxiliary tokenizer budgets; no changes to the pinned learner or its RNG."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import time

from .config import canonical_hash
from .storage import atomic_write_json,read_json


def own_token_counts(tokenizer,inputs):
    """Count actual unpadded model tokens on the already fixed visible text."""
    content=[];special=[]
    for start in range(0,len(inputs),512):
        texts=[row.text for row in inputs[start:start+512]]
        content.extend(len(ids) for ids in tokenizer(texts,add_special_tokens=False,
            truncation=False,verbose=False)['input_ids'])
        special.extend(len(ids) for ids in tokenizer(texts,add_special_tokens=True,
            truncation=False,verbose=False)['input_ids'])
    return {'rows':len(inputs),'content_tokens':sum(content),'tokens_with_specials':sum(special),
        'max_content_tokens':max(content,default=0),'max_tokens_with_specials':max(special,default=0),
        'includes_padding':False,'scope':'unique_visible_rows_not_optimizer_exposures'}


def record_token_budget(run,train_inputs,selection_inputs,assets,root: Path):
    from transformers import AutoTokenizer
    if run.model_id not in {'distilbert','modernbert'}:raise ValueError('Encoder tokenizer budget requires a declared encoder')
    if run.asset_hash!=assets['asset_hash']:raise ValueError('Tokenizer asset identity changed')
    root=Path(root).resolve();roles={'train':tuple(train_inputs),'select':tuple(selection_inputs)}
    identity={}
    for role,rows in roles.items():
        ids=[row.sample_id for row in rows]
        actual=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
        if len(set(ids))!=len(ids) or run.hparams['role_ids_hash'][role]!=actual:
            raise ValueError('Tokenizer budget role IDs differ from the authorized condition')
        if any(row.text_hash!=hashlib.sha256(row.text.encode()).hexdigest() for row in rows):
            raise ValueError('Tokenizer budget visible input bytes changed')
        identity[role]=sorted([row.sample_id,row.text_hash] for row in rows)
    if run.strict_no_val and roles['select']:raise ValueError('Strict run cannot inspect V for cost recording')
    source=assets['models'][run.model_id]
    if len(source.get('revision',''))!=40:raise ValueError('Immutable tokenizer revision is required')
    request=canonical_hash({'run':asdict(run),'visible_roles':identity,'tokenizer':source})
    path=root/'training/token-budgets'/f'{run.run_id}.json'
    if path.exists():
        previous=read_json(path)
        if previous.get('request_hash')!=request or previous.get('record_hash')!=canonical_hash({k:v for k,v in previous.items() if k!='record_hash'}):
            raise ValueError('Tokenizer budget cache identity/hash changed')
        return previous
    started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(source['directory'],local_files_only=True)
    counts={role:own_token_counts(tokenizer,rows) for role,rows in roles.items()}
    result={'run_id':run.run_id,'run_request_hash':canonical_hash(asdict(run)),'request_hash':request,
        'model_id':run.model_id,'input_hash':run.input_hash,'asset_hash':run.asset_hash,
        'tokenizer_revision':source['revision'],'roles':counts,'role_input_identity':identity,
        'auxiliary_tokenization_seconds':time.perf_counter()-started,
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'training_timing_changed':False,
        'includes_padding':False,'official_test_predictions_generated':False}
    result['record_hash']=canonical_hash(result);atomic_write_json(path,result,root)
    return result
