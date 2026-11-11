import copy
import unittest
from research_v3.config import canonical_hash


class SupplementalSafetyTests(unittest.TestCase):
    def test_runtime_paths_relocate_without_changing_locked_assets(self):
        from pathlib import Path
        from research_v3.supplemental import relocated_assets
        locked={'asset_hash':'fixed','files':[{'model_id':'distilbert','path':'assets/distilbert/config.json'}],
                'models':{'distilbert':{'directory':'C:\\old\\assets\\distilbert','revision':'a'*40}}}
        before=copy.deepcopy(locked)
        result=relocated_assets(locked,Path.cwd())
        self.assertEqual(locked,before)
        self.assertEqual(result['asset_hash'],'fixed')
        self.assertEqual(result['models']['distilbert']['directory'],str((Path.cwd()/'assets/distilbert').resolve()))

    def test_real_cpu_reload_preserves_cuda_bundle_bytes_and_scores(self):
        import tempfile
        import hashlib
        from pathlib import Path
        import joblib
        import numpy as np
        import torch
        from transformers import AutoTokenizer,DistilBertConfig,DistilBertForSequenceClassification
        from research_v3.supplemental import score_checkpoint_cpu
        from research_v3.types import VisibleInput
        from research_v3.storage import hash_file,atomic_write_json
        tokenizer=AutoTokenizer.from_pretrained('assets/distilbert',local_files_only=True)
        model=DistilBertForSequenceClassification(DistilBertConfig(vocab_size=len(tokenizer),n_layers=1,
                  n_heads=2,dim=8,hidden_dim=16,num_labels=2)).eval()
        text='Please explain this fee.'
        inputs=(VisibleInput('example',text,0,len(text),6,6,hashlib.sha256(text.encode()).hexdigest()),)
        with torch.inference_mode():expected=model(**tokenizer([text],return_tensors='pt')).logits.numpy()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);model.save_pretrained(output/'checkpoint')
            joblib.dump(dict(model_id='distilbert',task='single_label',label_count=2,features=None,
                hparams={'device':'cuda'},assets={'models':{'distilbert':{'directory':str(Path('assets/distilbert').resolve())}}},
                constants={}),output/'bundle.joblib')
            files={p.relative_to(output).as_posix():hash_file(p) for p in output.rglob('*') if p.is_file()}
            atomic_write_json(output/'status.json',dict(status='complete',reload_verified=True,files=files),output)
            before={p.relative_to(output).as_posix():hash_file(p) for p in output.rglob('*') if p.is_file()}
            actual,_=score_checkpoint_cpu(output,inputs)
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-5)
            self.assertEqual(before,{p.relative_to(output).as_posix():hash_file(p) for p in output.rglob('*') if p.is_file()})

    def test_real_manifest_keeps_sample_size_axis_separate_from_imbalance(self):
        import tempfile
        from pathlib import Path
        import shutil
        from research_v3.supplemental import prepare_manifest
        source=Path('research-v3')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for rel in ['training/overnight/run-plan.json','training/overnight/extension-run-plan.json',
                        'training/fixed-main-run-manifest.json','matrix/design_manifest.json']:
                (root/rel).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source/rel,root/rel)
            result=prepare_manifest(root)
            self.assertEqual(len(result['trajectory_sources']),24)
            self.assertEqual(len(result['fits']),24)
            axes={(r['dataset_id'],r['model_id'],r['training_rows']) for r in result['trajectory_sources']}
            self.assertEqual(len(axes),24)

    def test_cpu_clone_retains_roles_and_never_changes_source(self):
        from research_v3.supplemental import cpu_entry
        run = dict(run_id='old', condition_id='fixed', model_id='distilbert', training_seed=42,
                   hparams={'device':'cuda','lr':2e-5,'role_ids_hash':{'train':'T','select':'V','discovery':'empty'}},
                   input_hash='inputs', asset_hash='assets', code_hash='original', strict_no_val=False)
        entry = {'run':run,'run_id':'old','engine':'core','dataset_id':'banking77','condition_id':'fixed','model_id':'distilbert'}
        before = copy.deepcopy(entry)
        result = cpu_entry(entry, 'S1', max_epochs=10)
        self.assertEqual(entry, before)
        self.assertEqual(result['run']['hparams']['role_ids_hash'], run['hparams']['role_ids_hash'])
        self.assertEqual(result['run']['code_hash'], 'original')
        self.assertEqual(result['source_run_id'], 'old')
        self.assertEqual(result['run']['hparams']['device'], 'cpu')
        self.assertEqual(result['run_id'], canonical_hash({**result['run'], 'run_id':''}))
        run['strict_no_val']=True
        with self.assertRaises(ValueError):cpu_entry(entry,'S1')

    def test_trajectory_joins_by_id_and_rejects_changed_gold_or_missing_rows(self):
        from research_v3.supplemental import paired_trajectory
        small=[dict(sample_id='a',group_id='ga',dataset_labels=[1],prediction=[0]),
               dict(sample_id='b',group_id='gb',dataset_labels=[0],prediction=[0])]
        large=[dict(sample_id='b',group_id='gb',dataset_labels=[0],prediction=[1]),
               dict(sample_id='a',group_id='ga',dataset_labels=[1],prediction=[1])]
        result=paired_trajectory(small,large)
        self.assertEqual(result['counts'],{'persistent_error':0,'corrected':1,'new_error':1,'always_correct':0})
        with self.assertRaises(ValueError):paired_trajectory(small,large[:1])
        changed=copy.deepcopy(large);changed[0]['dataset_labels']=[1]
        with self.assertRaises(ValueError):paired_trajectory(small,changed)
        with self.assertRaises(ValueError):paired_trajectory(small+small[:1],large)

    def test_edit_structure_requires_matched_cue_and_noncue_substitutions(self):
        from research_v3.supplemental import check_draft
        pair={'original_text':'Please explain your exchange rate policy.', 'cue_words':['exchange','rate','your'],
              'intervention_text':'Please explain your conversion rate policy.',
              'control_text':'Please describe your exchange rate policy.'}
        self.assertEqual(check_draft(pair)['edit_count'],1)
        pair['control_text']='Please explain your exchange charge policy.'
        with self.assertRaises(ValueError):check_draft(pair)
        pair['control_text']='Please describe your exchange rate policy now.'
        with self.assertRaises(ValueError):check_draft(pair)


if __name__=='__main__':unittest.main()
