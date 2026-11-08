from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_v3.config import canonical_hash
from tests.research import test_training


class ConvergenceTrainingTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.convergence_training')
        except ModuleNotFoundError:
            self.fail('independent convergence learner is missing')

    def test_delayed_learning_is_not_stopped_before_minimum(self):
        s = self.api().StopState(min_epochs=5, patience=2, min_delta=.001)
        for epoch, value in enumerate([.01,.01,.01,.01,.01,.1,.1], 1):
            self.assertFalse(s.update(epoch,value))
        self.assertTrue(s.update(8,.1))
        self.assertEqual(s.best_epoch,6)

    def test_small_cumulative_gains_reset_patience_but_raw_best_is_retained(self):
        s=self.api().StopState(min_epochs=1,patience=3,min_delta=.001)
        self.assertFalse(s.update(1,.1000))
        self.assertFalse(s.update(2,.1004))
        self.assertFalse(s.update(3,.1008))
        self.assertFalse(s.update(4,.1012))
        self.assertEqual(s.bad_epochs,0)
        self.assertEqual(s.best_epoch,4)
        self.assertFalse(s.update(5,.1012))
        self.assertEqual(s.best_epoch,4)
        self.assertFalse(s.update(6,.1011))
        self.assertTrue(s.update(7,.1013))
        self.assertEqual(s.best_epoch,7)

    def test_stop_state_rejects_nonfinite_and_nonsequential_epoch(self):
        s=self.api().StopState(1,2,.0001)
        with self.assertRaises(ValueError):s.update(1,float('nan'))
        self.assertFalse(s.update(1,.1))
        with self.assertRaises(ValueError):s.update(3,.2)

    def test_warmup_has_no_total_epoch_dependency(self):
        api=self.api()
        self.assertEqual([api.lr_multiplier(i,4) for i in range(7)],[0,.25,.5,.75,1,1,1])
        self.assertEqual([api.lr_multiplier(i,0) for i in range(4)],[1,1,1,1])

    def test_runtime_guard_rejects_unavailable_gpu_and_excess_cpu_threads(self):
        api=self.api()
        from unittest.mock import patch
        with patch.dict('os.environ',{'SLURM_CPUS_PER_TASK':'4'}):
            api.validate_runtime('cpu',4)
            with self.assertRaises(ValueError):api.validate_runtime('cpu',8)
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):api.validate_runtime('cuda',4)

    def test_declared_fp32_scorer_never_enables_autocast(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        class Toy(torch.nn.Module):
            def forward(self,ids,lengths):return torch.ones((len(ids),2),dtype=torch.float32)
        features=SimpleNamespace(sequences=lambda rows:[[1,2] for _ in rows])
        with patch('torch.autocast',side_effect=AssertionError('undeclared mixed precision')):
            result=self.api().score_fp32(Toy(),'bilstm',(None,None),features,None,
                {'device':'cpu','micro_batch':1,'label_count':2},{1:False})
        self.assertEqual(result.dtype,np.float32)
        self.assertEqual(result[:,1].tolist(),[-30,-30])

    def test_resume_restores_dropout_rng_optimizer_and_schedule(self):
        import torch
        api=self.api()
        def new():
            torch.manual_seed(11)
            m=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.3))
            o=torch.optim.Adam(m.parameters(),lr=.01)
            sched=torch.optim.lr_scheduler.LambdaLR(o,lambda step:api.lr_multiplier(step,3))
            return m,o,sched
        def step(m,o,s):
            o.zero_grad();loss=m(torch.ones(3,2)).square().mean();loss.backward();o.step();s.step()
        with tempfile.TemporaryDirectory() as tmp:
            m,o,s=new()
            for _ in range(3):step(m,o,s)
            api.save_resume(Path(tmp)/'resume.pt',{'request_hash':'bound','model':m.state_dict(),
                'optimizer':o.state_dict(),'scheduler':s.state_dict(),'torch_rng':torch.get_rng_state()})
            for _ in range(4):step(m,o,s)
            expected={k:v.clone() for k,v in m.state_dict().items()}
            m2,o2,s2=new();saved=api.load_resume(Path(tmp)/'resume.pt','bound')
            m2.load_state_dict(saved['model']);o2.load_state_dict(saved['optimizer'])
            s2.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['torch_rng'])
            for _ in range(4):step(m2,o2,s2)
            for k,v in expected.items():self.assertTrue(torch.equal(v,m2.state_dict()[k]))
            with self.assertRaises(ValueError):api.load_resume(Path(tmp)/'resume.pt','other')

    def test_interruption_between_snapshot_and_pointer_preserves_previous_epoch(self):
        from unittest.mock import patch
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resume.pt'
            api.save_resume(path,{'request_hash':'same','epoch':1})
            with patch.object(api,'atomic_write_json',side_effect=OSError('simulated interruption')):
                with self.assertRaises(OSError):api.save_resume(path,{'request_hash':'same','epoch':2})
            self.assertEqual(api.load_resume(path,'same')['epoch'],1)

    def test_actual_micro_fit_saves_verified_best_and_distinct_stop_reason(self):
        api=self.api()
        from research_v3.train import score_saved
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,run=test_training.TrainingTests().fixture(root,model='bilstm',strict=False)
            rows=data.partitions['train']
            selected=tuple(replace(r,sample_id=r.sample_id.replace('train','select'),group_id='v'+r.group_id) for r in rows)
            data.partitions['select']=selected
            for train_row,val_row in zip(rows,selected):
                inputs[val_row.sample_id]=replace(inputs[train_row.sample_id],sample_id=val_row.sample_id)
            hp={**run.hparams,'max_epochs':4,'min_epochs':1,'patience':1,'min_delta':1.,
                'effective_batch':2,'warmup_updates':0,'cpu_threads':1,
                'role_ids_hash':{'train':api.ids_hash([r.sample_id for r in rows]),
                    'select':api.ids_hash([r.sample_id for r in selected]),'discovery':api.ids_hash([])}}
            run=replace(run,run_id='',hparams=hp,code_hash=api.convergence_code_hash())
            run=replace(run,run_id=canonical_hash(asdict(run)))
            out=root/'runs'/run.run_id
            result=api.fit_convergence({'run':asdict(run)},data,inputs,{},out,
                datetime.now(timezone.utc)+timedelta(minutes=2))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['stop_reason'],'patience')
            self.assertTrue(result['convergence_verified'])
            self.assertTrue(result['reload_verified'])
            self.assertEqual(result['epochs_completed'],2)
            self.assertEqual(score_saved(out,tuple(inputs.values())).shape,(4,2))
            self.assertTrue((out/'resume.manifest.json').is_file())
            self.assertEqual(result['selected_epoch'],max(result['history'],key=lambda h:h['validation']['macro_f1'])['epoch'])


if __name__=='__main__':unittest.main()
