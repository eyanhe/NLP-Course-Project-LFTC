"""Entirely synthetic temporary D packets; no real human identity is invented."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import research_v3  # Arrow before Torch on Windows.
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from research_v3.config import canonical_hash
from research_v3.inputs import make_visible_input
from research_v3.interventions import build_evidence_windows, pilot_candidate_packets
from research_v3.storage import atomic_write_json, hash_file, read_json
from research_v3.types import Example, PreparedDataset, RunSpec, VisibleInput


class PilotTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.pilots')
        except ModuleNotFoundError:self.fail('D pilot import/freeze/predict workflow is missing')

    def fixture(self,root,mechanism='M1',fit=True):
        from transformers import BertTokenizerFast
        dataset='banking77' if mechanism=='M1' else 'yelp'
        pilot='Bank_M1_SVM' if mechanism=='M1' else 'Yelp_M4_DistilBERT'
        node={'pilot_id':pilot,'dataset_id':dataset,'mechanism':mechanism}
        directory=root/'tokenizer';directory.mkdir()
        (directory/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ni\nwant\nto\nchange\nmy\ncard\npin\ntoday\ncode\nnow\nrefund\ntext\nspecial\n',encoding='utf-8')
        tok=BertTokenizerFast(vocab_file=str(directory/'vocab.txt'),do_lower_case=True);tok.save_pretrained(directory)
        assets={'asset_hash':'assets','models':{'distilbert':{'directory':str(directory),'revision':'a'*40}}}
        atomic_write_json(root/'assets/assets.lock.json',assets,root)
        train=(Example(dataset,'train/0','pin card',(0,),'train','t0'),
               Example(dataset,'train/1','refund now',(1,),'train','t1'))
        select=tuple(Example(dataset,f'V/{i}',r.text,r.labels,'validation',f'v{i}') for i,r in enumerate(train))
        source='I want to change my card PIN today' if mechanism=='M1' else ' '.join(['text']*350+['special']+['text']*49)
        discovery=tuple(Example(dataset,f'D/{i}',source,(0,),'train',f'synthetic-group-{i}') for i in range(55))
        data=PreparedDataset(dataset,'single_label',('first','second'),{
            'train':train,'select':select,'discovery':discovery,'test':()},'pool')
        inputs={r.sample_id:make_visible_input(r,tok) for r in (*train,*select,*discovery)}
        path=root/f'data/{dataset}/discovery.parquet';path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{k:v for k,v in asdict(r).items() if k!='fields'} for r in discovery]),path)
        prepared={'complete':True,'identity':{'asset_hash':'assets'},'datasets':{dataset:{
            'label_names':list(data.label_names),'partitions':{'discovery':{
                'file':path.relative_to(root).as_posix(),'sha256':hash_file(path)}}}}}
        prepared['prepared_hash']=canonical_hash(prepared)
        atomic_write_json(root/'data/prepared_manifest.json',prepared,root)
        design={'design_hash':'synthetic-design','discovery_pilot_nodes':[node],
                'selection_nodes':[{'node_id':f'select:{dataset}:tfidf_svm' if mechanism=='M1' else f'select:{dataset}:distilbert',
                    'anchor_condition_id':'condition','training_seed':42}]}
        atomic_write_json(root/'matrix/design_manifest.json',design,root)
        from research_v3.matrix import write_design_lock
        write_design_lock(root,{'design':'matrix/design_manifest.json'},
                          {'protocol_hash':'synthetic-protocol','design_hash':'synthetic-design'})
        packets=pilot_candidate_packets(data,inputs,mechanism,cue_map={0:['pin'],1:['refund']})
        directory=root/'training/discovery-review'/pilot;directory.mkdir(parents=True)
        with (directory/'candidates.jsonl').open('w',encoding='utf-8') as stream:
            for packet in packets:stream.write(json.dumps(packet)+'\n')
        manifest={'pilot_id':pilot,'dataset_id':dataset,'mechanism':mechanism,'pilot_n_required':50,
            'candidate_order_fixed_before_model_reveal':True,'model_predictions_in_packet':False,
            'selection_node_reference':f'select:{dataset}:tfidf_svm' if mechanism=='M1' else f'select:{dataset}:distilbert',
            'design_hash':'synthetic-design','prepared_hash':prepared['prepared_hash'],
            'candidate_file':'candidates.jsonl','candidate_file_sha256':hash_file(directory/'candidates.jsonl'),
            'source_partition':'discovery'}
        atomic_write_json(directory/'manifest.json',manifest,root)
        atomic_write_json(directory.parent/'manifest.json',{'pilots':[manifest]},root)
        reviewed=[]
        for packet in packets:
            pair=deepcopy(packet['pair_template'])
            if mechanism=='M1':
                pair.update(intervention_text='I want to change my card code today',
                            control_text='I want to change my card PIN now')
                review={'blind_to_predictions':True,'intervention_label_preserved':True,
                        'control_label_preserved':True,'control_natural':True,
                        'intervention_content_words_replaced':True,'control_content_words_replaced':True}
            else:
                encoded=tok(source,add_special_tokens=False,return_offsets_mapping=True)
                span=[source.index('special'),source.index('special')+7]
                pair['evidence_span']=span
                pair['windows']=build_evidence_windows(source,encoded['offset_mapping'],span,pair['sample_id'])
                review={'blind_to_predictions':True,'gold_full_supported':True,'intervention_evidence_valid':True,
                        'control_no_marked_evidence':True,'control_has_other_cues':False}
            pair['reviews']=[dict(review,reviewer_id=f'SYNTHETIC_TEST_REVIEWER_{i}',fixture_only=True) for i in [1,2]]
            reviewed.append(pair)
        run=RunSpec('synthetic-cpu-anchor','condition','tfidf_svm' if mechanism=='M1' else 'distilbert',None if mechanism=='M1' else 42,
                    {'device':'cpu','cpu_threads':1,'C':1,'arm':'random_unfrozen','epochs':10},'inputs','assets','code',False)
        atomic_write_json(root/'training/conditions/condition.json',{'dataset_id':dataset,
            'metadata':{'pool_manifest_hash':'pool'},'train_ids':['train/0','train/1']},root)
        if fit:
            if mechanism=='M1':
                from research_v3.train import fit_run
                fit_run(run,data,inputs,assets,root/'runs'/run.run_id)
            else:
                from research_v3.train import fit_run
                from transformers import DistilBertConfig,DistilBertForSequenceClassification
                cfg=DistilBertConfig(vocab_size=len(tok),max_position_embeddings=512,n_layers=1,n_heads=2,
                    dim=8,hidden_dim=12,num_labels=2,dropout=0.,attention_dropout=0.,seq_classif_dropout=0.)
                DistilBertForSequenceClassification(cfg).save_pretrained(assets['models']['distilbert']['directory'])
                fit_run(run,data,inputs,assets,root/'runs'/run.run_id)
        statuspath=root/'runs'/run.run_id/'status.json'
        selection={'design_hash':'synthetic-design','choices':{manifest['selection_node_reference']:{
            'status':'complete','selected_run_id':run.run_id,'dataset_id':dataset,
            'model_id':'tfidf_svm' if mechanism=='M1' else 'distilbert',
            'evidence_sha256':hash_file(statuspath) if fit else 'not-fitted'}},'discovery_pilots':{}}
        atomic_write_json(root/'selections.json',selection,root)
        return pilot,reviewed,packets,tok

    # Picking submitted order or >50 groups permits effect-dependent selection.
    def test_freeze_uses_first_fifty_valid_declared_groups_and_is_immutable(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root,fit=False)
            frozen=api.freeze_pilot_edits(root,pilot,list(reversed(pairs)))
            payload=read_json(root/frozen['edits_file'])
            self.assertEqual([r['pair']['sample_id'] for r in payload['pairs']],
                             [r['sample_id'] for r in packets[:50]])
            self.assertEqual(len(payload['pairs']),50)
            self.assertEqual(hash_file(root/frozen['edits_file']),frozen['edits_file_sha256'])
            self.assertFalse((root/'selection.lock.json').exists())
            self.assertFalse((root/'evaluation.freeze.json').exists())
            again=api.freeze_pilot_edits(root,pilot,pairs)
            self.assertEqual(again['edits_file_sha256'],frozen['edits_file_sha256'])
            pairs[0]['intervention_text']='I want to change my card code now'
            with self.assertRaisesRegex(ValueError,'frozen|changed|immutable'):
                api.freeze_pilot_edits(root,pilot,pairs)

    # Inventing reviewer defaults, accepting one review, or repeating a reviewer must fail.
    def test_empty_single_and_duplicate_review_imports_do_not_freeze(self):
        api=self.api()
        for kind in ['empty','single','duplicate']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root,fit=False)
                pairs=pairs[:50]
                if kind=='empty':pairs=[]
                elif kind=='single':pairs[0]['reviews']=pairs[0]['reviews'][:1]
                else:pairs[0]['reviews'][1]['reviewer_id']=pairs[0]['reviews'][0]['reviewer_id']
                with self.assertRaisesRegex((ValueError,PermissionError),'50|review|insufficient'):
                    api.freeze_pilot_edits(root,pilot,pairs)
                self.assertFalse((root/'training/discovery-pilots'/pilot/'edits.freeze.json').exists())

    def test_earlier_candidates_need_explicit_human_decisions_and_original_rejections_are_preserved(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,_,_=self.fixture(root,fit=False)
            with self.assertRaisesRegex(PermissionError,'earlier|order|review'):
                api.freeze_pilot_edits(root,pilot,pairs[5:])
            pairs[0]['reviews'][0]['control_natural']=False
            pairs[0]['rejection_reason']='SYNTHETIC FIXTURE: reviewer rejects unnatural matched control'
            frozen=api.freeze_pilot_edits(root,pilot,pairs)
            payload=read_json(root/frozen['edits_file'])
            self.assertEqual(payload['pairs'][0]['pair']['sample_id'],pairs[1]['sample_id'])
            self.assertEqual(payload['reviewed_pairs'][0],pairs[0])
            self.assertEqual(payload['rejected_reviewed_pairs'][0]['human_rejection_reason'],pairs[0]['rejection_reason'])
            self.assertEqual(api.freeze_pilot_edits(root,pilot,pairs),frozen)

    def test_real_pilot_edits_cannot_be_frozen_without_the_design_lock(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,_,_=self.fixture(root,fit=False)
            (root/'design.lock.json').unlink(missing_ok=True)
            with self.assertRaisesRegex(PermissionError,'design.*lock'):
                api.freeze_pilot_edits(root,pilot,pairs)

    # Source provenance cannot be supplied by the editor or laundered through packet fields.
    def test_changed_source_gold_group_and_cue_list_are_rejected(self):
        api=self.api()
        for field,value in [('original_text','I want to change your card PIN today'),
                            ('gold_labels',[1]),('group_id','other-group'),('cue_words',['card'])]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root,fit=False)
                pairs[0][field]=value
                with self.assertRaisesRegex(ValueError,'source|gold|group|cue|provenance'):
                    api.freeze_pilot_edits(root,pilot,pairs)

    # Scores must come from the authenticated selected saved model and feed the real freeze verifier.
    def test_actual_cpu_svm_predicts_frozen_edits_and_merges_only_when_called(self):
        api=self.api()
        from research_v3.freeze import _verify_pilot
        from research_v3.train import score_saved
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root)
            api.freeze_pilot_edits(root,pilot,pairs)
            before=(root/'selections.json').read_bytes()
            result=api.evaluate_pilot(root,pilot)
            self.assertEqual((root/'selections.json').read_bytes(),before)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['source_partition'],'discovery')
            payload=read_json(root/result['results_file']);self.assertEqual(len(payload['pairs']),50)
            record=payload['pairs'][0]
            self.assertIn('original_scores',record,'M1 must preserve the original input response')
            text=pairs[0]['intervention_text']
            visible=VisibleInput('synthetic-edited',text,0,len(text),8,8,'irrelevant-fixture-hash')
            actual=score_saved(root/'runs/synthetic-cpu-anchor',(visible,))[0]
            np.testing.assert_allclose(record['intervention_scores'],actual,rtol=0,atol=0)
            original_text=record['pair']['original_text']
            original=VisibleInput('synthetic-original',original_text,0,len(original_text),8,8,'fixture')
            np.testing.assert_allclose(record['original_scores'],score_saved(root/'runs/synthetic-cpu-anchor',(original,))[0],rtol=0,atol=0)
            files={};node=read_json(root/'matrix/design_manifest.json')['discovery_pilot_nodes'][0]
            checked=_verify_pilot(root,node,result,read_json(root/'selections.json')['choices'],files)
            self.assertEqual(checked['pilot_n'],50)
            api.merge_pilot_result(root,pilot)
            self.assertEqual(read_json(root/'selections.json')['discovery_pilots'][pilot]['results_sha256'],result['results_sha256'])
            self.assertFalse((root/'selection.lock.json').exists())

    # Frozen byte changes and selected checkpoint changes must stop scoring.
    def test_evaluation_rejects_changed_frozen_edits_and_checkpoint_bytes(self):
        api=self.api()
        for kind in ['edits','checkpoint','source']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root)
                frozen=api.freeze_pilot_edits(root,pilot,pairs)
                if kind=='edits':(root/frozen['edits_file']).write_text('{}',encoding='utf-8')
                elif kind=='checkpoint':(root/'runs/synthetic-cpu-anchor/bundle.joblib').write_bytes(b'changed')
                else:(root/'data/banking77/discovery.parquet').write_bytes(b'changed')
                with self.assertRaisesRegex((ValueError,PermissionError),'hash|changed|source|artifact'):
                    api.evaluate_pilot(root,pilot)
                self.assertFalse((root/'training/discovery-pilots'/pilot/'results.json').exists())

    # M4 score inputs must be exact real-token windows, not substituted snippets.
    def test_m4_authenticates_window_text_and_predicts_tiny_cpu_checkpoint(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root,'M4')
            wrong=deepcopy(pairs);wrong[0]['windows']['intervention']['text']='fabricated evidence'
            with self.assertRaisesRegex(ValueError,'window|source'):
                api.freeze_pilot_edits(root,pilot,wrong)
            api.freeze_pilot_edits(root,pilot,pairs)
            result=api.evaluate_pilot(root,pilot)
            self.assertEqual(result['status'],'complete')
            payload=read_json(root/result['results_file'])
            self.assertEqual(np.asarray(payload['pairs'][0]['intervention_scores']).shape,(2,))

    # A self-consistent packet template cannot overrule the actual D source.
    def test_rehashed_packet_template_cannot_replace_actual_original_text(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root,fit=False)
            packet_path=root/'training/discovery-review'/pilot/'candidates.jsonl'
            changed=[deepcopy(row) for row in packets]
            changed[0]['pair_template']['original_text']='I want to change your card PIN today'
            pairs[0].update(original_text='I want to change your card PIN today',
                intervention_text='I want to change your card code today',control_text='I want to change your card PIN now')
            packet_path.write_text(''.join(json.dumps(row)+'\n' for row in changed),encoding='utf-8')
            manifest_path=packet_path.parent/'manifest.json';manifest=read_json(manifest_path)
            manifest['candidate_file_sha256']=hash_file(packet_path)
            atomic_write_json(manifest_path,manifest,root)
            atomic_write_json(packet_path.parent.parent/'manifest.json',{'pilots':[manifest]},root)
            with self.assertRaisesRegex(ValueError,'source|provenance'):
                api.freeze_pilot_edits(root,pilot,pairs)

    # Existing result reuse must remain bound to the exact human-edits freeze.
    def test_rehashed_results_cannot_substitute_a_different_reviewed_edit(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root)
            api.freeze_pilot_edits(root,pilot,pairs);result=api.evaluate_pilot(root,pilot)
            path=root/result['results_file'];payload=read_json(path)
            row=payload['pairs'][0];row['pair']['intervention_text']='I want to change my card passcode today'
            row['edits_sha256']=canonical_hash(row['pair']);atomic_write_json(path,payload,root)
            result['results_sha256']=hash_file(path)
            atomic_write_json(path.parent/'result-manifest.json',result,root)
            with self.assertRaisesRegex(ValueError,'frozen|edits|pair'):
                api.evaluate_pilot(root,pilot)

    # A bank selection node must not silently accept another dataset's choice.
    def test_wrong_dataset_anchor_and_merging_after_global_lock_are_rejected(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,packets,tok=self.fixture(root)
            api.freeze_pilot_edits(root,pilot,pairs)
            path=root/'selections.json';selections=read_json(path)
            selections['choices']['select:banking77:tfidf_svm']['dataset_id']='yahoo_answers_topics'
            atomic_write_json(path,selections,root)
            with self.assertRaisesRegex(ValueError,'dataset|anchor|identity'):
                api.evaluate_pilot(root,pilot)
            selections['choices']['select:banking77:tfidf_svm']['dataset_id']='banking77'
            atomic_write_json(path,selections,root);api.evaluate_pilot(root,pilot)
            atomic_write_json(root/'selection.lock.json',{'synthetic_existing_lock':True},root)
            before=path.read_bytes()
            with self.assertRaisesRegex(PermissionError,'frozen|lock'):
                api.merge_pilot_result(root,pilot)
            self.assertEqual(path.read_bytes(),before)

    def test_global_selection_freeze_also_checks_result_against_original_edits_freeze(self):
        api=self.api()
        from research_v3.freeze import _verify_pilot
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,pairs,_,_=self.fixture(root)
            api.freeze_pilot_edits(root,pilot,pairs);result=api.evaluate_pilot(root,pilot)
            path=root/result['results_file'];payload=read_json(path)
            row=payload['pairs'][0];row['pair']['intervention_text']='I want to change my card passcode today'
            row['edits_sha256']=canonical_hash(row['pair']);atomic_write_json(path,payload,root)
            result['results_sha256']=hash_file(path)
            node=read_json(root/'matrix/design_manifest.json')['discovery_pilot_nodes'][0]
            with self.assertRaisesRegex(ValueError,'frozen|edits|pair'):
                _verify_pilot(root,node,result,read_json(root/'selections.json')['choices'],{})


if __name__=='__main__':unittest.main()
