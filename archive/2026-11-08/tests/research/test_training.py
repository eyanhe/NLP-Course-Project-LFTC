"""Training must consume exact T IDs and reload only matching saved artifacts."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from research_v3.config import canonical_hash
from research_v3.types import Example,PreparedDataset,RunSpec,VisibleInput


class RestrictedPartitions(dict):
    def __getitem__(self,key):
        if key in {'select','discovery','test'}:
            raise AssertionError(f'strict learner accessed forbidden {key} partition')
        return super().__getitem__(key)


class TrainingTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.train')
        except ModuleNotFoundError:self.fail('research training implementation is missing')

    def fixture(self,root,model='tfidf_lr',strict=True):
        rows=(Example('fixture','train/0','hidden raw invisible words',(0,),'train','g0'),
              Example('fixture','train/1','different hidden raw words',(1,),'train','g1'))
        partitions={'train':rows,'select':(),'discovery':(),'test':(),'quarantine':()}
        data=PreparedDataset('fixture','single_label',('negative','positive'),
                             RestrictedPartitions(partitions) if strict else partitions,'pool')
        inputs={sid:VisibleInput(sid,text,0,len(text),2,2,sid) for sid,text in
                [('train/0','awful bad'),('train/1','great good')]}
        (root/'matrix').mkdir()
        (root/'matrix/design_manifest.json').write_text(json.dumps({'conditions':[
            {'condition_id':'condition','dataset_id':'fixture','train_ids':['train/0','train/1'],
             'metadata':{'pool_manifest_hash':'pool'}}]}),encoding='utf-8')
        hp={'C':1,'ngram_range':[1,2],'device':'cpu','epochs':10,'effective_batch':2,'micro_batch':1,
            'hidden_per_direction':4,'embedding_initialization':{'vector_size':8,'epochs':1},
            'lr':.001,'dropout':.3}
        run=RunSpec('fixture-run','condition',model,42 if model=='bilstm' else None,hp,'input','assets','code',strict)
        return data,inputs,run

    def test_strict_training_ignores_forbidden_holdouts_and_reloads_identical_scores(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,inputs,run=self.fixture(root)
            output=root/'runs'/run.run_id
            result=api.fit_run(run,data,inputs,{},output)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['training_unique_rows'],2)
            self.assertEqual(result['holdout_labels_accessed'],0)
            self.assertTrue(result['reload_verified'])
            self.assertIn('resources',result,'actual training memory samples are missing')
            self.assertGreater(result['resources']['peak_rss_sampled_bytes'],0)
            self.assertGreaterEqual(result['resources']['sample_count'],2)
            scores=api.score_saved(output,tuple(inputs.values()))
            self.assertEqual(scores.shape,(2,2))
            self.assertEqual(scores.argmax(axis=1).tolist(),[0,1])

    def test_changed_request_cannot_reuse_a_complete_directory(self):
        from dataclasses import replace
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,inputs,run=self.fixture(root)
            output=root/'runs'/run.run_id;api.fit_run(run,data,inputs,{},output)
            changed=replace(run,hparams={**run.hparams,'C':10})
            with self.assertRaisesRegex(ValueError,'identity|request|changed'):
                api.fit_run(changed,data,inputs,{},output)

    def test_modified_artifact_cannot_be_reloaded(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,inputs,run=self.fixture(root)
            output=root/'runs'/run.run_id;result=api.fit_run(run,data,inputs,{},output)
            (output/'bundle.joblib').write_bytes(b'changed saved bytes')
            with self.assertRaisesRegex(ValueError,'hash|changed'):
                api.score_saved(output,tuple(inputs.values()))

    def test_strict_training_rejects_an_anchor_selection_reference(self):
        from dataclasses import replace
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,inputs,run=self.fixture(root)
            run=replace(run,hparams={**run.hparams,'selected_by':'anchor'})
            with self.assertRaisesRegex(ValueError,'strict|anchor|selection'):
                api.fit_run(run,data,inputs,{},root/'runs'/run.run_id)

    def test_effective_batch_survives_microbatch_halving_and_last_partial_group(self):
        api=self.api()
        groups=api.accumulation_groups(np.arange(11),effective_batch=4,micro_batch=2)
        self.assertEqual([[len(chunk) for chunk in group] for group in groups],[[2,2],[2,2],[2,1]])
        smaller=api.accumulation_groups(np.arange(11),effective_batch=4,micro_batch=1)
        self.assertEqual([sum(len(chunk) for chunk in group) for group in smaller],[4,4,3])

    def test_bilstm_strict_epochs_and_saved_scores_without_holdout_access(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,inputs,run=self.fixture(root,model='bilstm')
            output=root/'runs'/run.run_id
            result=api.fit_run(run,data,inputs,{},output)
            self.assertEqual(result['epochs_completed'],10)
            self.assertEqual(result['selected_epoch'],10)
            self.assertEqual(result['holdout_labels_accessed'],0)
            self.assertTrue(result['reload_verified'])
            self.assertEqual(api.score_saved(output,tuple(inputs.values())).shape,(2,2))

    def test_wrong_checkpoint_is_detected_against_actual_presave_scores(self):
        from unittest.mock import patch
        from safetensors.torch import load_file,save_file
        api=self.api();original=api._checkpoint
        def bad_saved_checkpoint(model,model_id,output):
            original(model,model_id,output)
            path=output/'checkpoint/model.safetensors'
            state=load_file(str(path));state['classifier.bias']+=10
            save_file(state,str(path))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,inputs,run=self.fixture(root,model='bilstm')
            with patch.object(api,'_checkpoint',bad_saved_checkpoint):
                with self.assertRaisesRegex(ValueError,'[Rr]eload|score|Saved'):
                    api.fit_run(run,data,inputs,{},root/'runs'/run.run_id)


if __name__=='__main__':unittest.main()
