"""Synthetic temporary Q frames/humans; global freeze alone is fixture-stubbed."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import research_v3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from research_v3.config import canonical_hash
from research_v3.audit import draw_audit
from research_v3.inputs import make_visible_input
from research_v3.interventions import build_evidence_windows
from research_v3.storage import atomic_write_json, hash_file, read_json
from research_v3.types import Example, PreparedDataset, RunSpec


class ConfirmationTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.confirmation')
        except ModuleNotFoundError:self.fail('confirmation preparation/freeze/F6 emitter is missing')

    def fixture(self,root,mechanism='M1'):
        from transformers import BertTokenizerFast
        from research_v3.train import fit_run
        root=Path(root);dataset='banking77' if mechanism=='M1' else 'yelp'
        pilot='Bank_M1_SVM' if mechanism=='M1' else 'Yelp_M4_DistilBERT'
        tokdir=root/'tokenizer';tokdir.mkdir()
        (tokdir/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ni\nwant\nto\nchange\nmy\ncard\npin\ntoday\ncode\nnow\nrefund\ntext\nspecial\n',encoding='utf-8')
        tok=BertTokenizerFast(vocab_file=str(tokdir/'vocab.txt'),do_lower_case=True);tok.save_pretrained(tokdir)
        assets={'asset_hash':'assets','models':{'distilbert':{'directory':str(tokdir),'revision':'a'*40}}}
        atomic_write_json(root/'assets/assets.lock.json',assets,root)
        source='I want to change my card PIN today' if mechanism=='M1' else ' '.join(['text']*350+['special']+['text']*49)
        group_ids=sorted(f'g{i}' for i in range(205));rng=np.random.default_rng(7301)
        drawn={group_ids[int(i)] for i in rng.choice(205,size=200,replace=False)}
        remainder=sorted(set(group_ids)-drawn);cross_a,discovery_group=remainder[:2]
        allmeta={};refs={}
        for ds in ['yelp','go_emotions','banking77','yahoo_answers_topics']:
            n=205 if ds==dataset else 1
            rows=[Example(ds,f'{ds}/Q/{i}',source,(0,),'test',
                          f'g{i}' if ds==dataset else (cross_a if ds=='go_emotions' else f'{ds}-audit')) for i in range(n)]
            directory=root/'data'/ds;directory.mkdir(parents=True)
            path=directory/'test.parquet'
            pq.write_table(pa.Table.from_pylist([{k:v for k,v in asdict(r).items() if k!='fields'} for r in rows]),path)
            discovery=directory/'discovery.parquet'
            pq.write_table(pa.Table.from_pylist([{'sample_id':f'{ds}/D/0','group_id':discovery_group,'labels':[0]}]),discovery)
            inputs=[dict(asdict(make_visible_input(r,tok)),partition='test') for r in rows]
            inp=directory/'inputs.parquet';pq.write_table(pa.Table.from_pylist(inputs),inp)
            allmeta[ds]={'dataset_id':ds,'task':'single_label','label_names':['first','second'],'manifest_hash':'pool',
                         'inputs':{'file':inp.relative_to(root).as_posix(),'file_sha256':hash_file(inp),'input_hash':'inputs'},
                         'partitions':{'test':{'file':path.relative_to(root).as_posix(),'sha256':hash_file(path),'rows':len(rows)},
                                       'discovery':{'file':discovery.relative_to(root).as_posix(),'sha256':hash_file(discovery)}}}
            audit=root/'audit'/f'{ds}.json'
            drawn=draw_audit(PreparedDataset(ds,'single_label',('first','second'),{'test':tuple(rows)},'pool'),
                             'audit',200,7301,set())
            atomic_write_json(audit,{'dataset_id':ds,'purpose':'audit','selection_policy':'uniform_groups_then_uniform_row',
                                    'model_outcomes_used_for_sampling':False,'sampling_seed':7301,
                                    'records':list(drawn)},root)
            refs[ds]={'file':audit.relative_to(root).as_posix(),'sha256':hash_file(audit)}
        train=tuple(Example(dataset,f'T/{i}','pin' if i%2==0 else 'refund',(i%2,),'train',f't{i}') for i in range(6000))
        select=(Example(dataset,'V/0','pin',(0,),'validation','v0'),Example(dataset,'V/1','refund',(1,),'validation','v1'))
        data=PreparedDataset(dataset,'single_label',('first','second'),{'train':train,'select':select},'pool')
        traininputs={r.sample_id:make_visible_input(r,tok) for r in (*train,*select)}
        trainpath=root/'data'/dataset/'train.parquet';pq.write_table(pa.Table.from_pylist([
            {k:v for k,v in asdict(r).items() if k!='fields'} for r in train]),trainpath)
        allmeta[dataset]['partitions']['train']={'file':trainpath.relative_to(root).as_posix(),'sha256':hash_file(trainpath)}
        ipath=root/allmeta[dataset]['inputs']['file'];qt=pq.read_table(ipath).to_pylist()
        pq.write_table(pa.Table.from_pylist(qt+[dict(asdict(v),partition='train') for k,v in traininputs.items() if k.startswith('T/')]),ipath)
        allmeta[dataset]['inputs']['file_sha256']=hash_file(ipath)
        prepared={'complete':True,'identity':{'asset_hash':'assets'},'datasets':allmeta}
        prepared['prepared_hash']=canonical_hash(prepared);atomic_write_json(root/'data/prepared_manifest.json',prepared,root)
        condition={'condition_id':'anchor','dataset_id':dataset,'data_seed':2026,'experiment_id':'E1',
                   'metadata':{'size':6000,'pool_manifest_hash':'pool'},'train_ids':[r.sample_id for r in train],
                   'training_set_hash':hashlib.sha256('\n'.join(sorted(r.sample_id for r in train)).encode()).hexdigest()}
        design={'design_hash':'synthetic-design','conditions':[condition],
                'selection_nodes':[{'node_id':f'select:{dataset}:'+('tfidf_svm' if mechanism=='M1' else 'distilbert'),
                                    'anchor_condition_id':'anchor'}],
                'discovery_pilot_nodes':[{'pilot_id':pilot,'dataset_id':dataset,'mechanism':mechanism}]}
        atomic_write_json(root/'matrix/design_manifest.json',design,root)
        run=RunSpec('synthetic-anchor','anchor','tfidf_svm',None,{'device':'cpu','C':1,'cpu_threads':1,
                    'role_ids_hash':{'train':condition['training_set_hash']}},'inputs','assets','code',False)
        # M4 uses an authenticated fixture SVM payload behind the declared tiny
        # Distil checkpoint; score_saved is explicitly fixture-stubbed for M4.
        fit_run(run,data,traininputs,assets,root/'runs'/run.run_id)
        statuspath=root/'runs'/run.run_id/'status.json';status=read_json(statuspath)
        if mechanism=='M4':
            status['run']['model_id']='distilbert';status['run']['training_seed']=42
            atomic_write_json(statuspath,status,root)
        request=canonical_hash(status['run'])
        manifest={'status':'resolved','design_hash':'synthetic-design','runs':[{
            'run_id':run.run_id,'artifact_id':run.run_id,'planned_run_id':'planned','dataset_id':dataset,
            'run':status['run'],'expected_model_request_hash':request}]}
        atomic_write_json(root/'run_manifest.json',manifest,root)
        model='tfidf_svm' if mechanism=='M1' else 'distilbert'
        choices={f'select:{dataset}:{model}':{'status':'complete','selected_run_id':run.run_id,'dataset_id':dataset,
                                           'model_id':model,'evidence_sha256':hash_file(statuspath)}}
        plans={pilot:{'pilot_n':50,'n_planned':400,'checkpoint_run_id':run.run_id,'human_reviews_present':True,
                      'source_partition':'discovery'}}
        atomic_write_json(root/'selections.json',{'design_hash':'synthetic-design','choices':choices},root)
        selection={'lock_hash':'synthetic-selection','pilots':plans}
        views={f'eval:{pilot}:{arm}':{'dataset_id':dataset,'model_id':model,'data_seed':2026,
            'training_seed':None if mechanism=='M1' else 42,'checkpoint_run_id':run.run_id,
            'evaluation_id':f'eval:{pilot}:{arm}','n_planned':400,'base_group_namespace':f'paired:{pilot}'}
               for arm in ['intervention','control']}
        atomic_write_json(root/'contrasts.json',{'design_hash':'synthetic-design','runs':views},root)
        return pilot,refs,tok,selection

    @contextmanager
    def frozen(self,api,selection):
        def verified(stage,root):
            return selection if stage=='selection' else {'lock_hash':'synthetic-evaluation','design_hash':'synthetic-design'}
        with patch.object(api,'verify_freeze',side_effect=verified):yield

    def reviewed(self,payload,tok,mechanism='M1'):
        result=[]
        for candidate in payload['candidates']:
            pair=deepcopy(candidate['pair_template'])
            if mechanism=='M1':
                pair.update(intervention_text='I want to change my card code today',
                            control_text='I want to change my card PIN now')
                review={'blind_to_predictions':True,'intervention_label_preserved':True,'control_label_preserved':True,
                        'control_natural':True,'intervention_content_words_replaced':True,'control_content_words_replaced':True}
            else:
                source=pair['original_text'];span=[source.index('special'),source.index('special')+7]
                offsets=tok(source,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                pair['evidence_span']=span;pair['windows']=build_evidence_windows(source,offsets,span,pair['sample_id'])
                review={'blind_to_predictions':True,'gold_full_supported':True,'intervention_evidence_valid':True,
                        'control_no_marked_evidence':True,'control_has_other_cues':False}
            pair['reviews']=[dict(review,reviewer_id=f'SYNTHETIC_REVIEWER_{i}',fixture_only=True) for i in [1,2]]
            result.append(pair)
        return result

    def test_each_public_api_requires_real_evaluation_freeze_before_q_or_models(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            for name,args in [('prepare_confirmation',{'audit_frames':{}}),('freeze_confirmation_edits',{'reviewed_pairs':[]}),
                              ('predict_confirmation',{}),('verify_confirmation',{}),
                              ('verify_confirmation_audit_frame',{'actual_global_frame':{}})]:
                with self.subTest(name=name),self.assertRaisesRegex(PermissionError,'evaluation'):
                    getattr(api,name)(tmp,'Bank_M1_SVM',**args)

    def test_prepare_excludes_cross_dataset_a_and_discovery_groups_with_blank_reviews(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                payload=read_json(root/manifest['candidates_file'])
                groups=[r['group_id'] for r in payload['candidates']]
                audit_groups={r['group_id'] for ref in refs.values() for r in read_json(root/ref['file'])['records']}
                source_groups={r['group_id'] for r in pq.read_table(root/'data/banking77/test.parquet').to_pylist()}
                d_groups={r['group_id'] for r in pq.read_table(root/'data/banking77/discovery.parquet').to_pylist()}
                self.assertEqual(set(groups),source_groups-audit_groups-d_groups)
                self.assertEqual(len(groups),3)
                self.assertEqual(len(groups),len(set(groups)))
                self.assertTrue(manifest['underpowered']);self.assertEqual(manifest['n_planned'],400)
                self.assertEqual(manifest['seed'],7301)
                self.assertGreaterEqual(len(payload['rejections']),202)
                self.assertTrue(all(not r['pair_template']['reviews'] for r in payload['candidates']))
                self.assertTrue(all(t['reviewer_id'] is None for r in payload['candidates'] for t in r['review_templates']))
                again=api.prepare_confirmation(root,pilot,audit_frames=refs)
                self.assertEqual(again,manifest)

    def test_freeze_requires_all_decisions_and_two_actual_reviews_then_is_immutable(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                pairs=self.reviewed(read_json(root/manifest['candidates_file']),tok)
                with self.assertRaisesRegex(PermissionError,'unreviewed|decision|pending'):
                    api.freeze_confirmation_edits(root,pilot,pairs[1:])
                invalid=deepcopy(pairs);invalid[0]['reviews']=invalid[0]['reviews'][:1]
                with self.assertRaisesRegex((ValueError,PermissionError),'review'):
                    api.freeze_confirmation_edits(root,pilot,invalid)
                frozen=api.freeze_confirmation_edits(root,pilot,list(reversed(pairs)))
                self.assertEqual(frozen['n_actual'],3);self.assertTrue(frozen['underpowered'])
                self.assertEqual(frozen,api.freeze_confirmation_edits(root,pilot,pairs))
                changed=deepcopy(pairs);changed[0]['intervention_text']='I want to change my card passcode today'
                with self.assertRaisesRegex(ValueError,'immutable|changed|frozen'):
                    api.freeze_confirmation_edits(root,pilot,changed)

    def test_fixed_cpu_checkpoint_emits_exact_f6_views_and_preserves_originals(self):
        api=self.api()
        from research_v3.train import score_saved
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                pairs=self.reviewed(read_json(root/manifest['candidates_file']),tok)
                api.freeze_confirmation_edits(root,pilot,pairs)
                result=api.predict_confirmation(root,pilot)
                self.assertEqual(result['n_actual'],3);self.assertTrue(result['underpowered'])
                self.assertIn('mcnemar_sensitivity',result)
                self.assertEqual(result['mcnemar_sensitivity']['n'],3)
                self.assertEqual(result['mcnemar_sensitivity']['role'],'sensitivity_only')
                self.assertGreaterEqual(result['checkpoint_to_scores_seconds'],0.)
                self.assertEqual(set(result['views']),{f'eval:{pilot}:intervention',f'eval:{pilot}:control'})
                a=pq.read_table(root/result['views'][f'eval:{pilot}:intervention']['file']).to_pylist()
                b=pq.read_table(root/result['views'][f'eval:{pilot}:control']['file']).to_pylist()
                self.assertEqual([(r['sample_id'],r['group_id'],r['labels']) for r in a],
                                 [(r['sample_id'],r['group_id'],r['labels']) for r in b])
                self.assertTrue(all(r['run_id']==f'eval:{pilot}:intervention' for r in a))
                self.assertEqual(len(pq.read_table(root/result['original']['file'])),3)
                self.assertEqual(result,api.predict_confirmation(root,pilot))
                row=a[0];pair=next(p for p in pairs if p['sample_id']==row['sample_id'])
                inp=make_visible_input(Example('banking77',pair['sample_id'],pair['intervention_text'],(0,),'test',pair['group_id']),tok)
                np.testing.assert_allclose(row['scores'],score_saved(root/'runs/synthetic-anchor',(inp,))[0],rtol=0,atol=0)

    def test_rejection_is_recorded_without_replacement_or_relaxed_eligibility(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                pairs=self.reviewed(read_json(root/manifest['candidates_file']),tok)
                pairs[0]={'sample_id':pairs[0]['sample_id'],'decision':'reject','reason':'No natural valid matched control',
                          'reviews':pairs[0]['reviews']}
                frozen=api.freeze_confirmation_edits(root,pilot,pairs)
                self.assertEqual(frozen['n_actual'],2);self.assertTrue(frozen['underpowered'])
                payload=read_json(root/frozen['edits_file'])
                self.assertEqual(len(payload['rejections']),1)
                self.assertEqual(payload['rejections'][0]['reason'],'No natural valid matched control')

    def test_source_gold_cues_and_m4_windows_are_authenticated(self):
        api=self.api()
        for mechanism in ['M1','M4']:
            with self.subTest(mechanism=mechanism),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);pilot,refs,tok,selection=self.fixture(root,mechanism)
                with self.frozen(api,selection):
                    manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                    pairs=self.reviewed(read_json(root/manifest['candidates_file']),tok,mechanism)
                    wrong=deepcopy(pairs)
                    if mechanism=='M1':wrong[0]['cue_words']=['today']
                    else:wrong[0]['windows']['intervention']['text']='fabricated'
                    with self.assertRaisesRegex(ValueError,'cue|window|source'):
                        api.freeze_confirmation_edits(root,pilot,wrong)
                    wrong=deepcopy(pairs);wrong[0]['gold_labels']=[1]
                    with self.assertRaisesRegex(ValueError,'gold|source|provenance'):
                        api.freeze_confirmation_edits(root,pilot,wrong)
                    api.freeze_confirmation_edits(root,pilot,pairs)
                    if mechanism=='M4':
                        with patch('research_v3.train.score_saved',return_value=np.tile([.7,.3],(3,1))):
                            result=api.predict_confirmation(root,pilot)
                        self.assertNotIn('original',result)

    def test_missing_or_changed_audit_reference_and_changed_checkpoint_block(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                with self.assertRaisesRegex((ValueError,PermissionError),'audit'):
                    api.prepare_confirmation(root,pilot,audit_frames={})
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                pairs=self.reviewed(read_json(root/manifest['candidates_file']),tok)
                api.freeze_confirmation_edits(root,pilot,pairs)
                (root/'runs/synthetic-anchor/bundle.joblib').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'hash|changed|artifact'):
                    api.predict_confirmation(root,pilot)

    def test_rehashed_result_cannot_replace_frozen_pair_on_cache_reuse(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                pairs=self.reviewed(read_json(root/manifest['candidates_file']),tok)
                api.freeze_confirmation_edits(root,pilot,pairs);result=api.predict_confirmation(root,pilot)
                path=root/result['results_file'];payload=read_json(path)
                payload['pairs'][0]['pair']['intervention_text']='I want to change my card passcode today'
                payload['pairs'][0]['edits_sha256']=canonical_hash(payload['pairs'][0]['pair'])
                atomic_write_json(path,payload,root)
                result['results_sha256']=hash_file(path)
                atomic_write_json(path.parent/'result-manifest.json',result,root)
                with self.assertRaisesRegex(ValueError,'frozen|edits|pair|identity'):
                    api.predict_confirmation(root,pilot)

    def test_read_only_verifier_authenticates_rows_and_recomputes_power_summary(self):
        api=self.api()
        self.assertTrue(hasattr(api,'verify_confirmation'),'read-only confirmation verifier is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                pairs=self.reviewed(read_json(root/manifest['candidates_file']),tok)
                api.freeze_confirmation_edits(root,pilot,pairs);result=api.predict_confirmation(root,pilot)
                before={p.relative_to(root).as_posix():hash_file(p) for p in (root/'confirmation').rglob('*') if p.is_file()}
                self.assertEqual(result,api.verify_confirmation(root,pilot))
                after={p.relative_to(root).as_posix():hash_file(p) for p in (root/'confirmation').rglob('*') if p.is_file()}
                self.assertEqual(before,after)
                result['underpowered']=False;result['mechanism_evidence_sufficient']=True
                atomic_write_json(root/'confirmation'/pilot/'result-manifest.json',result,root)
                with self.assertRaisesRegex(ValueError,'summary|power|count'):
                    api.verify_confirmation(root,pilot)

    def test_no_eligible_pairs_does_not_claim_human_reviews_or_generate_responses(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            prepared=read_json(root/'data/prepared_manifest.json')
            part=prepared['datasets']['banking77']['partitions']['discovery']
            path=root/part['file'];rows=pq.read_table(path).to_pylist()
            groups={r['group_id'] for r in pq.read_table(root/'data/banking77/test.parquet').to_pylist()}
            excluded={r['group_id'] for ref in refs.values() for r in read_json(root/ref['file'])['records']}
            rows.extend({'sample_id':f'D/exclude/{g}','group_id':g,'labels':[0]} for g in sorted(groups-excluded))
            pq.write_table(pa.Table.from_pylist(rows),path);part['sha256']=hash_file(path)
            prepared['prepared_hash']=canonical_hash({k:v for k,v in prepared.items() if k!='prepared_hash'})
            atomic_write_json(root/'data/prepared_manifest.json',prepared,root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                self.assertEqual(manifest['eligible_groups'],0)
                frozen=api.freeze_confirmation_edits(root,pilot,[])
                self.assertFalse(frozen['real_human_reviews_verified'])
                self.assertTrue(frozen['underpowered'])
                with self.assertRaisesRegex(PermissionError,'No valid|pending'):
                    api.predict_confirmation(root,pilot)
                self.assertFalse((root/'confirmation'/pilot/'result-manifest.json').exists())

    def global_audit_frame(self,root,refs):
        return {'records':[r for ds in sorted(refs) for r in read_json(root/refs[ds]['file'])['records']],
                'seeds':{ds:7301 for ds in refs},'created_at_utc':'2026-11-07T00:00:00+00:00',
                'evaluation_freeze_hash':'synthetic-evaluation','guide_freeze_sha256':'a'*64}

    def test_alternate_valid_200_audit_groups_and_nonpredeclared_seed_are_refused(self):
        api=self.api()
        for mutation in ['alternate_valid_frame','other_seed']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
                path=root/refs['banking77']['file'];packet=read_json(path)
                if mutation=='other_seed':packet['sampling_seed']=7302
                else:
                    existing={r['sample_id'] for r in packet['records']}
                    replacement=next(r for r in pq.read_table(root/'data/banking77/test.parquet').to_pylist()
                                     if r['sample_id'] not in existing)
                    packet['records'][0].update(sample_id=replacement['sample_id'],group_id=replacement['group_id'],
                                               labels=replacement['labels'])
                atomic_write_json(path,packet,root);refs['banking77']['sha256']=hash_file(path)
                with self.frozen(api,selection),self.assertRaisesRegex(ValueError,'audit|Audit|seed|draw'):
                    api.prepare_confirmation(root,pilot,audit_frames=refs)

    def test_actual_global_audit_frame_must_match_same_bound_fixed_draw_and_seed(self):
        api=self.api()
        self.assertTrue(hasattr(api,'verify_confirmation_audit_frame'),'global A-frame binding verifier is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot,refs,tok,selection=self.fixture(root)
            with self.frozen(api,selection):
                manifest=api.prepare_confirmation(root,pilot,audit_frames=refs)
                frame=self.global_audit_frame(root,refs)
                summary=api.verify_confirmation_audit_frame(root,pilot,frame)
                self.assertTrue(summary['binding_verified'])
                self.assertEqual(summary['actual_global_frame_hash'],canonical_hash(frame))
                self.assertEqual(summary['confirmation_audit_frames'],refs)
                self.assertTrue(all(r['sampling_seed']==7301 for r in summary['datasets'].values()))
                for mutation in ['other_valid_A','other_seed','other_freeze','missing_dataset']:
                    wrong=deepcopy(frame)
                    if mutation=='other_seed':wrong['seeds']['banking77']=7302
                    elif mutation=='other_freeze':wrong['evaluation_freeze_hash']='different'
                    elif mutation=='missing_dataset':wrong['records']=[r for r in wrong['records'] if r['dataset_id']!='yelp']
                    else:
                        existing={r['sample_id'] for r in frame['records'] if r['dataset_id']=='banking77'}
                        replacement=next(r for r in pq.read_table(root/'data/banking77/test.parquet').to_pylist()
                                         if r['sample_id'] not in existing)
                        row=next(r for r in wrong['records'] if r['dataset_id']=='banking77')
                        row.update(sample_id=replacement['sample_id'],group_id=replacement['group_id'],labels=replacement['labels'])
                    with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'audit|Audit|frame|seed|freeze'):
                        api.verify_confirmation_audit_frame(root,pilot,wrong)


if __name__=='__main__':unittest.main()
