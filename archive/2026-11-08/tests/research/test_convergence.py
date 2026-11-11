from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.storage import atomic_write_json


class ConvergenceTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.convergence')
        except ModuleNotFoundError:self.fail('convergence dispatcher is missing')

    def test_declared_grid_and_device_changes_preserve_source_roles(self):
        api=self.api();root=Path(__file__).resolve().parents[2]/'research-v3'
        from research_v3.storage import read_json
        proposal=read_json(root/'training/convergence-v1/proposal.json')
        entries=api.build_candidates(proposal,distil_device='cpu')
        self.assertEqual(len(entries),18)
        self.assertEqual(len({r['run_id'] for r in entries}),18)
        sources={b['source_run_id']:b for b in proposal['source_bindings']}
        for entry in entries:
            source=sources[entry['source_run_id']]['source_run']
            self.assertEqual(source['condition_id'],entry['run']['condition_id'])
            self.assertEqual(source['hparams']['role_ids_hash'],entry['run']['hparams']['role_ids_hash'])
        gpu=api.build_candidates(proposal,distil_device='cuda')
        self.assertTrue(all(e['run']['hparams']['device']=='cuda' and e['run']['hparams']['cpu_threads']==4
            for e in gpu if e['model_id']=='distilbert'))

    def test_selection_rejects_unconverged_or_missing_candidates(self):
        api=self.api()
        records=[{'status':'complete','reload_verified':True,'convergence_verified':True,
            'validation':{'macro_f1':v},'run':{'model_id':'bilstm','training_seed':42,'hparams':{'lr':lr}}}
            for v,lr in [(.5,.003),(.5,.001),(.3,.0003)]]
        self.assertEqual(api.select_recipes(records,3,'macro_f1')['run']['hparams']['lr'],.001)
        records[0]['convergence_verified']=False
        with self.assertRaises(PermissionError):api.select_recipes(records,3,'macro_f1')
        with self.assertRaises(PermissionError):api.select_recipes(records[:1],3,'macro_f1')

    def test_shared_claim_is_atomic_and_cutoff_prevents_new_fits(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest={'start_cutoff_utc':'2026-11-08T01:00:00+00:00','candidates':[
                {'run_id':str(i),'run':{'hparams':{'device':'cpu'}}} for i in range(12)]}
            now=datetime(2026,11,7,20,tzinfo=timezone.utc)
            def claim(i):return api.claim_next(root,str(i),now,manifest=manifest,device='cpu')
            with ThreadPoolExecutor(max_workers=12) as pool:claimed=list(pool.map(claim,range(12)))
            self.assertEqual(len({x['run_id'] for x in claimed}),12)
            self.assertIsNone(claim(99))
            with self.assertRaises(PermissionError):api.claim_next(root,'late',datetime(2026,11,8,2,tzinfo=timezone.utc),manifest=manifest,device='cpu')

    def test_only_proven_dead_slurm_owner_can_be_reclaimed(self):
        api=self.api()
        from research_v3.storage import read_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest={'start_cutoff_utc':'2026-11-08T01:00:00+00:00',
                'candidates':[{'run_id':'one','model_id':'bilstm','run':{'hparams':{'device':'cpu'}}}]}
            now=datetime(2026,11,7,20,tzinfo=timezone.utc)
            api.claim_next(root,'first',now,manifest=manifest)
            path=root/api.REL/'claims/one.json';claim=read_json(path);claim['job_id']='1234';atomic_write_json(path,claim,root)
            self.assertIsNone(api.claim_next(root,'next',now,manifest=manifest,job_state=lambda job:'RUNNING'))
            self.assertEqual(api.claim_next(root,'next',now,manifest=manifest,job_state=lambda job:'FAILED')['run_id'],'one')

    def test_unfinished_claim_publication_is_conservatively_skipped(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/api.REL/'claims';folder.mkdir(parents=True)
            (folder/'one.json').write_text('{',encoding='utf-8')
            manifest={'start_cutoff_utc':'2026-11-08T01:00:00+00:00',
                'candidates':[{'run_id':'one','model_id':'bilstm','run':{'hparams':{'device':'cpu'}}}]}
            self.assertIsNone(api.claim_next(root,'other',datetime(2026,11,7,20,tzinfo=timezone.utc),manifest=manifest))


if __name__=='__main__':unittest.main()
