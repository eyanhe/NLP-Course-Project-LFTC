"""Fixed hyperparameters resolve configuration without six candidate fits."""
from dataclasses import asdict
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.config import PROJECT_ROOT,load_protocol
from research_v3.matrix import candidate_grid,training_policy
from research_v3.orchestrator import _spec
from research_v3.storage import atomic_write_json,hash_file
from research_v3.train import learner_code_hash


class FixedPolicyTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.fixed_policy')
        except ModuleNotFoundError:self.fail('user fixed-hyperparameter policy is missing')

    def fixture(self):
        p=load_protocol(PROJECT_ROOT/'configs/research-v3.json')
        meta={'dataset_id':'fixture','task':'single_label','label_names':['a','b'],
              'inputs':{'input_hash':'input'}}
        design={'design_hash':'design','protocol_hash':p['_protocol_hash'],
                'conditions':[{'condition_id':'C','dataset_id':'fixture','training_set_hash':'T'}],
                'selection_nodes':[{'node_id':f'select:fixture:{m}','model_id':m,'dataset_id':'fixture',
                    'anchor_condition_id':'C','training_seed':42,'candidates':candidate_grid(m,p)}
                    for m in ['tfidf_svm','distilbert']]}
        return p,design,{'datasets':{'fixture':meta}}

    def test_fixed_defaults_are_one_config_per_family_and_reuse_existing_request_identity(self):
        api=self.api();p,d,prepared=self.fixture()
        policy=api.build_fixed_policy(p,d,prepared,{'fixture':'V'},'assets',learner_code_hash(),'user fixes hyperparameters')
        self.assertEqual(len(policy['nodes']),2)
        row=policy['nodes']['select:fixture:tfidf_svm']
        self.assertEqual(row['hparams'],{'C':1,'ngram_range':[1,2]})
        hp={**training_policy('tfidf_svm',p),**row['hparams'],
            'role_ids_hash':{'train':'T','select':'V','discovery':api.EMPTY_IDS_HASH},'target_task':'single_label'}
        expected=_spec(p,prepared['datasets']['fixture'],'C','tfidf_svm',hp,'assets',learner_code_hash())
        self.assertEqual(row['run'],asdict(expected))
        self.assertEqual(policy['nodes']['select:fixture:distilbert']['hparams'],{'lr':2e-5,'weight_decay':.01})

    def test_missing_checkpoint_does_not_trigger_grid_training_or_fake_completion(self):
        api=self.api();p,d,prepared=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);policy=api.build_fixed_policy(p,d,prepared,{'fixture':'V'},'assets',learner_code_hash(),'user fixes hyperparameters')
            atomic_write_json(root/'training/fixed-hyperparameters.json',policy,root)
            choices=api.fixed_choices(root,policy)
            choice=choices['select:fixture:distilbert']
            self.assertEqual(choice['status'],'complete')  # Configuration, not trained weights.
            self.assertEqual(choice['checkpoint_status'],'pending')
            self.assertFalse((root/'runs').exists())
            from research_v3.freeze import verify_anchor_choice
            verify_anchor_choice(root,d['selection_nodes'][1],choice,{})

    def test_fixed_choice_cannot_substitute_a_different_grid_candidate(self):
        api=self.api();p,d,prepared=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);policy=api.build_fixed_policy(p,d,prepared,{'fixture':'V'},'assets',learner_code_hash(),'user fixes hyperparameters')
            atomic_write_json(root/'training/fixed-hyperparameters.json',policy,root)
            choice=api.fixed_choices(root,policy)['select:fixture:tfidf_svm'];choice['hparams']['C']=100
            from research_v3.freeze import verify_anchor_choice
            with self.assertRaisesRegex(ValueError,'fixed|configuration|policy'):
                verify_anchor_choice(root,d['selection_nodes'][0],choice,{})


if __name__=='__main__':unittest.main()
