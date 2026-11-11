"""Execution permission is separate from the immutable experimental protocol."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

import psutil

from .matrix import verify_design_lock
from .storage import read_json,hash_file,bounded_path


def authorize_phase(p: dict,root: Path,authorization: Path,phase: str) -> dict:
    root=Path(root)
    permit=read_json(authorization)
    if permit.get('protocol_hash')!=p['_protocol_hash']:
        raise ValueError('Execution authorization protocol identity changed')
    lock=verify_design_lock(root)
    if permit.get('design_lock_hash')!=lock['lock_hash']:
        raise ValueError('Execution authorization design-lock identity changed')
    if not permit.get('user_instruction') or not permit.get('training_enabled'):
        raise PermissionError('Explicit training execution authorization is required')
    if permit.get('test_predictions_enabled'):
        raise PermissionError('This training authorization cannot enable official-test predictions')
    if phase not in permit.get('allowed_training_phases',[]):
        raise PermissionError(f'Training phase is not authorized: {phase}')
    if phase=='anchors':
        g0=root/'training/engineering-smoke.json'
        if not g0.exists() or not read_json(g0).get('complete'):
            raise PermissionError('Actual E0 engineering fits and reloads must pass before anchor search')
        smoke=read_json(g0)
        cases=smoke.get('cases',[])
        if len(cases)!=15 or len({(r['dataset_id'],r['model_id']) for r in cases})!=15:
            raise PermissionError('E0 requires 15 actual engineering case artifacts')
        if smoke.get('design_hash')!=lock['bindings']['design_hash']:
            raise PermissionError('E0 engineering artifacts refer to a different design')
        from .train import learner_code_hash
        if smoke.get('learner_code_hash')!=learner_code_hash():
            raise PermissionError('E0 engineering artifacts use different learner code')
        for case in cases:
            path=bounded_path(root/case['run_status'],root)
            if not path.exists() or hash_file(path)!=case['run_status_sha256']:
                raise PermissionError('E0 engineering case status artifact changed')
            status=read_json(path)
            if status['status']!='complete' or not status.get('reload_verified'):
                raise PermissionError('E0 case checkpoint reload is not complete')
            for relative,expected in status['files'].items():
                target=bounded_path(path.parent/relative,path.parent)
                if not target.exists() or hash_file(target)!=expected:
                    raise PermissionError('E0 case checkpoint artifact changed')
            scores=bounded_path(root/case['score_rows']['path'],root)
            if not scores.exists() or hash_file(scores)!=case['score_rows']['sha256']:
                raise PermissionError('E0 engineering score artifact changed')
    elif phase in {'main','mechanisms'}:
        if not (root/'selection.lock.json').exists():
            raise PermissionError('selection.lock requires actual anchor choices and three real D pilots')
        from .freeze import verify_freeze
        verify_freeze('selection',root)
    return permit


@contextmanager
def run_lease(output: Path,request_hash: str):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'.lease.json'
    token=uuid.uuid4().hex
    owner={'pid':os.getpid(),'process_created':psutil.Process().create_time(),
           'request_hash':request_hash,'token':token,'started_at':time.time()}
    for attempt in range(3):
        try:
            with path.open('x',encoding='utf-8') as stream:
                json.dump(owner,stream);stream.flush();os.fsync(stream.fileno())
            break
        except FileExistsError:
            previous=read_json(path)
            try:
                process=psutil.Process(int(previous['pid']))
                live=process.is_running() and abs(process.create_time()-float(previous['process_created']))<.01
            except (psutil.NoSuchProcess,psutil.ZombieProcess):
                live=False
            if live:raise RuntimeError(f'Exclusive run lease is already running in PID {previous["pid"]}')
            if previous.get('request_hash')!=request_hash:
                raise ValueError('Stale lease belongs to a different run request identity')
            path.replace(output/f'.lease.history.{time.time_ns()}.json')
    else:
        raise RuntimeError('Could not acquire exclusive run lease')
    try:
        yield owner
    finally:
        if path.exists() and read_json(path).get('token')==token:
            path.unlink()
