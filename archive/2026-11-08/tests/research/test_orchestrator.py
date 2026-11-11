"""Concrete engineering plans bind role IDs and never read raw hidden text."""
import importlib
from pathlib import Path
import tempfile
import unittest
from research_v3.config import load_protocol
from research_v3.storage import read_json

ROOT=Path(__file__).resolve().parents[2]


class OrchestratorTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.orchestrator')
        except ModuleNotFoundError:self.fail('training phase orchestration is missing')

    def fixture(self):
        p=load_protocol(ROOT/'configs/research-v3.json')
        p['experiments']['E0']={'training_unique_rows':2,'validation_rows':1,'evaluation_rows':1}
        packet={'dataset_id':'banking77','training_ids':['T/0','T/1'],'selection_ids':['V/0'],
                'engineering_evaluation_ids':['D/0'],'label_count':2,'label_names':['a','b']}
        meta={'dataset_id':'banking77','task':'single_label','label_names':['a','b'],
              'manifest_hash':'pool','inputs':{'input_hash':'visible'}}
        return p,packet,meta

    def test_engineering_specs_bind_actual_ids_and_neural_seed(self):
        api=self.api();p,packet,meta=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            linear=api.engineering_spec(p,packet,'tfidf_lr',meta,'assets','code',root)
            neural=api.engineering_spec(p,packet,'bilstm',meta,'assets','code',root)
            self.assertIsNone(linear.training_seed);self.assertEqual(neural.training_seed,42)
            self.assertEqual(neural.hparams['effective_batch'],64)
            self.assertEqual(linear.condition_id,neural.condition_id)
            condition=read_json(root/'training/conditions'/f'{linear.condition_id}.json')
            self.assertEqual(condition['train_ids'],['T/0','T/1'])
            self.assertFalse(linear.strict_no_val)

    def test_role_overlap_is_rejected_before_fitting(self):
        api=self.api();p,packet,meta=self.fixture()
        packet['selection_ids']=['T/0']
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'overlap|role'):
                api.engineering_spec(p,packet,'tfidf_lr',meta,'assets','code',Path(temp))

    def test_anchor_ties_use_declared_candidate_order(self):
        records=[{'candidate_index':0,'validation':{'macro_f1':.6}},
                 {'candidate_index':1,'validation':{'macro_f1':.7}},
                 {'candidate_index':2,'validation':{'macro_f1':.7}}]
        selected=self.api().choose_candidate(records,'single_label')
        self.assertEqual(selected['candidate_index'],1)

    def test_go_anchor_uses_ap_instead_of_operating_f1(self):
        records=[{'candidate_index':0,'validation':{'macro_ap':.8,'macro_f1':.1}},
                 {'candidate_index':1,'validation':{'macro_ap':.7,'macro_f1':.9}}]
        selected=self.api().choose_candidate(records,'multilabel')
        self.assertEqual(selected['candidate_index'],0)


if __name__=='__main__':unittest.main()
