"""Reports verify bytes and underlying records; missing research remains pending."""
import csv
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict
import hashlib

from research_v3.config import PROJECT_ROOT,canonical_hash
from research_v3.freeze import freeze_stage
from research_v3.matrix import write_design_lock
from research_v3.storage import atomic_write_json,hash_file,read_json


def save(root,path,value):
    atomic_write_json(root/path,value,root)


class ReportTests(unittest.TestCase):
    def api(self):
        try: return importlib.import_module('research_v3.report')
        except ModuleNotFoundError: self.fail('Evidence-driven report implementation is missing')

    def prediction_fixture(self,root,dataset='yelp',model_id='tfidf_svm'):
        """Explicit synthetic saved evidence, never fitting a model or predicting Q."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        model=root/'runs/r/model.bin';model.parent.mkdir(parents=True);model.write_bytes(b'synthetic model bytes, not a trained checkpoint')
        run={'run_id':'r','model_id':model_id,'condition_id':'c','training_seed':None,'input_hash':'input','asset_hash':'synthetic-assets'}
        if model_id=='modernbert':
            from tests.research.test_costs import CostTests
            from research_v3.types import RunSpec
            _,token_train,token_select,token_dir=CostTests().fixture(root)
            ih=lambda rows:hashlib.sha256('\n'.join(sorted(r.sample_id for r in rows)).encode()).hexdigest()
            actual_run=RunSpec('r','c','modernbert',42,{'role_ids_hash':{'train':ih(token_train),'select':ih(token_select)}},
                               'input','synthetic-assets','synthetic-code',False)
            run=asdict(actual_run)
            assets={'asset_hash':'synthetic-assets','models':{'modernbert':{'directory':str(token_dir),'revision':'a'*40}}}
            save(root,'assets/assets.lock.json',assets)
        save(root,'runs/r/status.json',{'status':'complete','reload_verified':True,'reload_max_absolute_difference':0.,
             'run_id':'r','run':run,'files':{'model.bin':hash_file(model)},'score_kind':'probability'})
        design={'design_hash':'synthetic-design','conditions':[{'condition_id':'c','dataset_id':dataset,'experiment_id':'E1',
                 'data_seed':2026,'metadata':{'size':2}}], 'physical_runs':[{'run_id':'planned'}],
                'selection_nodes':[],'discovery_pilot_nodes':[]}
        save(root,'matrix/design_manifest.json',design)
        (root/'data').mkdir()
        pq.write_table(pa.Table.from_pylist([{'sample_id':str(i),'labels':[i],'group_id':'g'+str(i)} for i in (0,1)]),root/'data/q.parquet')
        pq.write_table(pa.Table.from_pylist([{'sample_id':str(i),'partition':'test','text_hash':'visible'+str(i)} for i in (0,1)]),root/'data/inputs.parquet')
        prepared={'datasets':{dataset:{'task':'single_label','label_names':['negative','positive'],
                  'inputs':{'input_hash':'input','file':'data/inputs.parquet'},
                  'partitions':{'test':{'rows':2,'ap_mask':[0,1],'file':'data/q.parquet'}}}}}
        save(root,'data/prepared_manifest.json',prepared)
        write_design_lock(root,{'design':'matrix/design_manifest.json','prepared':'data/prepared_manifest.json',
                          'q':'data/q.parquet','inputs':'data/inputs.parquet'},
                          {'protocol_hash':'synthetic-protocol','design_hash':'synthetic-design'})
        manifest={'status':'resolved','unresolved_nodes':[],'runs':[{'run_id':'r','artifact_id':'r',
                  'planned_run_id':'planned','dataset_id':dataset,'model_id':model_id,'condition_id':'c','run':run,
                  'expected_model_request_hash':canonical_hash(run)}]}
        save(root,'run_manifest.json',manifest)
        source={p.relative_to(PROJECT_ROOT).as_posix():hash_file(p) for p in (PROJECT_ROOT/'research_v3').glob('*.py')}
        selection={'stage':'selection','protocol_hash':'synthetic-protocol','design_hash':'synthetic-design',
            'files':{'runs':{'path':'run_manifest.json','sha256':hash_file(root/'run_manifest.json')}},
            'source_files':source,'source_inventory':sorted(source),'actual_human_pilots_verified':True}
        selection['lock_hash']=canonical_hash(selection);save(root,'selection.lock.json',selection)
        save(root,'contrasts.json',{'design_hash':'synthetic-design','items':[{'contrast_id':'synthetic'}]})
        save(root,'thresholds.json',{'design_hash':'synthetic-design','runs':{'r':{'operative':None}}})
        if model_id=='modernbert':
            from research_v3.costs import record_token_budget
            record_token_budget(actual_run,token_train,token_select,assets,root)
        lock=freeze_stage('evaluation',root,{})
        rows=[{'run_id':'r','sample_id':str(i),'group_id':'g'+str(i),'labels':[i],
               'prediction':[i],'scores':[.9,.1] if i==0 else [.1,.9],
               'score_kind':'probability','input_hash':'input'} for i in (0,1)]
        path=root/'predictions/r.parquet';path.parent.mkdir();pq.write_table(pa.Table.from_pylist(rows),path)
        from research_v3.evaluate import _metrics,_read_predictions
        metrics=_metrics(_read_predictions(path),prepared['datasets'][dataset])
        request=canonical_hash({'run':run,'evaluation_freeze_hash':lock['lock_hash'],
                               'rows':[(str(i),'g'+str(i),(i,),'visible'+str(i)) for i in (0,1)]})
        save(root,'predictions/r.manifest.json',{'run_id':'r','dataset_id':dataset,'rows':2,'request_hash':request,
            'evaluation_freeze_hash':lock['lock_hash'],'file_sha256':hash_file(path),
            'metrics':metrics,'metrics_hash':canonical_hash(metrics),'score_dtype':'float64'})
        return path

    def test_empty_root_lists_specific_missing_gates_and_stays_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=self.api().verify_research(Path(tmp),'full_research')
            self.assertFalse(out['complete'])
            self.assertFalse(out['training_complete'])
            self.assertEqual(set(out['gates']),{'G0','G1','G2','G3','G4','G5'})
            self.assertIn('selection.lock.json',{x['path'] for x in out['missing']})
            self.assertIn('evaluation.freeze.json',{x['path'] for x in out['missing']})

    def test_unverified_completion_and_human_booleans_cannot_override_real_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root,'completion.json',{'complete':True,'scope':'full_research','human_review':True})
            save(root,'analysis/research-evidence.json',{'complete':True,'human_review':True,'all_runs_complete':True})
            out=self.api().verify_research(root,'full_research')
            self.assertFalse(out['complete'])
            self.assertFalse(out['gates']['G4']['complete'])

    def test_metric_without_saved_prediction_rows_is_not_research_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root,'results.json',{'yelp':{'macro_f1':1.}})
            out=self.api().verify_research(root,'full_research')
            self.assertFalse(out['complete'])
            self.assertEqual(out['results'],[])
            self.assertEqual(out['prediction_access'],'blocked_until_evaluation_freeze')

    def test_byte_bound_saved_prediction_metrics_reconstruct_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prediction_fixture(root)
            out=self.api().verify_research(root,'full_research')
            self.assertFalse(out['complete'])  # Deliberately lacks real E0, anchors, people and stats.
            self.assertEqual(out['results'][0]['metrics']['macro_f1'],1.)
            self.assertEqual(out['results'][0]['test_exposure_status'],'historical_Q_seen_exploratory_replication')
            self.assertTrue(any(e['path']=='predictions/r.parquet' and e['sha256'] for e in out['evidence']))

    def test_changed_prediction_bytes_or_metric_receipt_cannot_be_reported(self):
        for mutation in ('bytes','metric'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path=self.prediction_fixture(root)
                if mutation=='bytes':path.write_bytes(b'changed prediction bytes')
                else:
                    receipt=read_json(root/'predictions/r.manifest.json');receipt['metrics']['macro_f1']=.2
                    receipt['metrics_hash']=canonical_hash(receipt['metrics'])
                    save(root,'predictions/r.manifest.json',receipt)
                out=self.api().verify_research(root,'full_research')
                self.assertFalse(out['complete'])
                self.assertEqual(out['results'],[])
                self.assertTrue(out['gates']['G3']['failures'])

    def test_missing_planned_training_run_cannot_be_filled_by_manifest_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prediction_fixture(root)
            design=read_json(root/'matrix/design_manifest.json');design['physical_runs'].append({'run_id':'missing'})
            save(root,'matrix/design_manifest.json',design)
            out=self.api().verify_research(root,'training')
            self.assertFalse(out['complete'])
            self.assertFalse(out['training_complete'])

    def test_reports_are_bilingual_task_separated_and_write_only_the_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'state';root.mkdir();self.prediction_fixture(root)
            # Another task is declared; no invented score is written for it.
            prepared=read_json(root/'data/prepared_manifest.json')
            before={p.relative_to(root).as_posix():hash_file(p) for p in root.rglob('*') if p.is_file()}
            output=Path(tmp)/'delivery'
            out=self.api().build_report(root,output)
            self.assertFalse(out['complete'])
            self.assertEqual({p.name for p in output.iterdir()},
                {'research-report-zh.md','research-report-en.md','evidence-map.csv','completion.json'})
            self.assertEqual(before,{p.relative_to(root).as_posix():hash_file(p) for p in root.rglob('*') if p.is_file()})
            en=(output/'research-report-en.md').read_text(encoding='utf-8')
            zh=(output/'research-report-zh.md').read_text(encoding='utf-8')
            self.assertIn('pending',en.lower());self.assertIn('unresolved',en)
            self.assertIn('historical_Q_seen_exploratory_replication',en)
            self.assertIn('未完成',zh)
            self.assertIn('No cross-task metric average',en)
            with (output/'evidence-map.csv').open(encoding='utf-8-sig',newline='') as stream:
                self.assertTrue(list(csv.DictReader(stream)))
            self.assertEqual(read_json(output/'completion.json')['complete'],False)

    def test_unknown_scope_is_an_error_not_a_weaker_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'scope'):
                self.api().verify_research(Path(tmp),'smoke')

    def test_training_scope_does_not_open_optional_official_prediction_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.prediction_fixture(root);path.write_bytes(b'not a parquet')
            out=self.api().verify_research(root,'training')
            self.assertEqual(out['prediction_access'],'not_requested_for_training_scope')
            self.assertFalse(any(f['check'].startswith('predictions:') for f in out['gates']['G3']['failures']))

    def audit_fixture(self,root):
        from research_v3.audit import draw_audit
        from research_v3.types import Example,PreparedDataset,PredictionRow
        prepared={'datasets':{}};predictions={};records=[];reviews=[];seeds={}
        for ds in ('yelp','go_emotions','banking77','yahoo_answers_topics'):
            examples=tuple(Example(ds,str(i),'synthetic human packet',(i%2,),'test',ds+str(i)) for i in range(200))
            data=PreparedDataset(ds,'single_label',('a','b'),{'test':examples},'synthetic')
            prepared['datasets'][ds]={'task':'single_label','label_names':['a','b']}
            predictions[ds]=tuple(PredictionRow('fixture',r.sample_id,r.group_id,r.labels,r.labels,(.8,.2),'probability','input') for r in examples)
            seeds[ds]=2026
            packet=draw_audit(data,'audit',200,2026,set());records.extend(packet)
            for row in packet:
                for person in ('synthetic_reviewer_A','synthetic_reviewer_B'):
                    reviews.append({**row['annotation_template'],'reviewer_id':person,'review_status':'double_review',
                                    'gold_status':'supported'})
        (root/'audit').mkdir(parents=True,exist_ok=True)
        (root/'audit/guide.md').write_text('Synthetic fixture annotation guide.',encoding='utf-8')
        save(root,'audit/guide.freeze.json',{'guide_sha256':hash_file(root/'audit/guide.md'),
             'frozen_at_utc':'2026-01-01T00:00:00+00:00'})
        save(root,'audit/frame.json',{'records':records,'seeds':seeds,'created_at_utc':'2026-01-02T00:00:00+00:00',
             'evaluation_freeze_hash':'synthetic','guide_freeze_sha256':hash_file(root/'audit/guide.freeze.json')})
        save(root,'audit/reviews.json',{'records':reviews,'human_review':True,'phase':'blind_first_pass'})
        return prepared,predictions,{'audit':{name:{'path':path,'sha256':hash_file(root/path)} for name,path in
            [('frame','audit/frame.json'),('reviews','audit/reviews.json'),('guide','audit/guide.md'),('guide_freeze','audit/guide.freeze.json')]}}

    def test_800_synthetic_audit_cases_require_1600_distinct_original_review_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,predictions,index=self.audit_fixture(root)
            api=self.api();out=api._audit(api._Evidence(root),index,prepared,predictions,{'lock_hash':'synthetic'})
            self.assertEqual(sum(row['samples'] for row in out['datasets']),800)
            self.assertEqual(sum(row['independent_review_records'] for row in out['datasets']),1600)
            review=read_json(root/'audit/reviews.json');review['records'].pop();save(root,'audit/reviews.json',review)
            index['audit']['reviews']['sha256']=hash_file(root/'audit/reviews.json')
            with self.assertRaisesRegex((PermissionError,ValueError),'review|Double'):
                api._audit(api._Evidence(root),index,prepared,predictions,{'lock_hash':'synthetic'})

    def test_supported_cause_without_reviewed_intervention_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.prediction_fixture(root)
            save(root,'analysis/claims.json',{'claims':[{'claim_id':'reason','kind':'interventional','status':'supported',
                'text':'Synthetic unsupported mechanism claim.','contrast_id':'invented',
                'reviewed_sample_ids':['yelp:0'],'evidence':[{'path':'predictions/r.parquet','sha256':hash_file(path)}]}]})
            save(root,'analysis/research-evidence.json',{'claims':{'path':'analysis/claims.json','sha256':hash_file(root/'analysis/claims.json')}})
            out=self.api().verify_research(root,'full_research')
            self.assertFalse(out['complete']);self.assertEqual(out['claims'][0]['status'],'unresolved')
            self.assertTrue(any(f['check'].startswith('reviewed_claim:') for f in out['gates']['G5']['failures']))

    def test_byte_bound_related_work_is_rendered_without_claiming_other_gates_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'state';root.mkdir();self.prediction_fixture(root)
            path=root/'analysis/related-work.csv';path.parent.mkdir()
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['citation','url','relevance','difference']);writer.writeheader()
                writer.writerows({'citation':'Synthetic reference '+str(i),'url':'https://example.org/'+str(i),
                                 'relevance':'Synthetic context','difference':'Synthetic comparison'} for i in range(5))
            save(root,'analysis/research-evidence.json',{'related_work':{'path':'analysis/related-work.csv','sha256':hash_file(path)}})
            output=Path(tmp)/'delivery';self.api().build_report(root,output)
            self.assertIn('Synthetic reference 0',(output/'research-report-en.md').read_text(encoding='utf-8'))

    def test_modern_token_budget_is_read_from_the_actual_freeze_verifier_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prediction_fixture(root,model_id='modernbert')
            out=self.api().verify_research(root,'training')
            budget=out['training']['costs'][0]['own_token_budget']
            self.assertIsNotNone(budget)
            self.assertEqual(budget['roles']['train']['content_tokens'],2)
            self.assertEqual(out['gates']['G3']['checks']['modern_token_budget:r']['status'],'verified')

    def test_review_disagreement_requires_adjudication_bound_to_both_original_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,predictions,index=self.audit_fixture(root)
            review=read_json(root/'audit/reviews.json');review['records'][-1]['gold_status']='ambiguous'
            save(root,'audit/reviews.json',review);index['audit']['reviews']['sha256']=hash_file(root/'audit/reviews.json')
            target=review['records'][-1]
            originals=[r for r in review['records'] if (r['dataset_id'],r['sample_id'])==(target['dataset_id'],target['sample_id'])]
            adjudication={'dataset_id':target['dataset_id'],'sample_id':target['sample_id'],
                'group_id':target['group_id'],'adjudicator_id':'synthetic_adjudicator',
                'review_record_hashes':sorted(canonical_hash(r) for r in originals),
                'gold_status':'ambiguous','phenomena':[],'rationale':'Synthetic preserved disagreement fixture.'}
            save(root,'audit/adjudications.json',{'records':[adjudication]})
            index['audit']['adjudications']={'path':'audit/adjudications.json','sha256':hash_file(root/'audit/adjudications.json')}
            api=self.api()
            try:out=api._audit(api._Evidence(root),index,prepared,predictions,{'lock_hash':'synthetic'})
            except PermissionError as exc:self.fail(f'Byte-bound human adjudication must resolve a preserved disagreement: {exc}')
            self.assertEqual(out['adjudicated_cases'],1)
            adjudication['review_record_hashes'][0]='not-an-original-review'
            save(root,'audit/adjudications.json',{'records':[adjudication]})
            index['audit']['adjudications']['sha256']=hash_file(root/'audit/adjudications.json')
            with self.assertRaisesRegex(ValueError,'adjudication|original'):
                api._audit(api._Evidence(root),index,prepared,predictions,{'lock_hash':'synthetic'})

    def test_two_named_review_records_with_unchecked_gold_remain_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,predictions,index=self.audit_fixture(root)
            payload=read_json(root/'audit/reviews.json')
            for record in payload['records']:record['gold_status']='not_checked'
            save(root,'audit/reviews.json',payload);index['audit']['reviews']['sha256']=hash_file(root/'audit/reviews.json')
            api=self.api()
            with self.assertRaisesRegex(PermissionError,'gold|review'):
                api._audit(api._Evidence(root),index,prepared,predictions,{'lock_hash':'synthetic'})

    def test_each_statistics_family_receives_only_its_required_run_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);families=['F1','F2','F3','F4','F5','F6','AUX_M2','AUX_M3','AUX_M5']
            manifest={'contrasts':[],'estimates':[]};index={'statistics':[]};prediction_refs={}
            for family in families:
                ids=[family+'_left',family+'_right']
                manifest['contrasts'].append({'contrast_id':family,'family':family,'components':[{'estimate_id':family,'coefficient':1}]})
                manifest['estimates'].append({'estimate_id':family,'draws':[{'data_seed':2026,'left_runs':ids[:1],'right_runs':ids[1:]}]})
                for run in ids:
                    path=root/f'{run}.parquet'
                    pq.write_table(pa.Table.from_pylist([{'run_id':run,'sample_id':'x','group_id':'g',
                        'labels':[0],'prediction':[0],'scores':[.9,.1],'score_kind':'probability','input_hash':'i'}]),path)
                    prediction_refs[run]={'path':path.name,'sha256':hash_file(path)}
                result={'status':'ok','critical':0.,'contrasts':{family:{'estimate':0.}}}
                save(root,f'{family}.json',result)
            save(root,'contrasts.json',manifest)
            for family in families:
                index['statistics'].append({'family':family,'path':family+'.json','sha256':hash_file(root/(family+'.json')),
                    'contrast_file_sha256':hash_file(root/'contrasts.json')})
            # Unrelated input is deliberately unreadable as parquet: it must stay unopened.
            unrelated=root/'unrelated.parquet';unrelated.write_bytes(b'unrelated invalid parquet')
            prediction_refs['unrelated']={'path':unrelated.name,'sha256':hash_file(unrelated)}
            def bounded_bootstrap(rows,selected,replicates,seed):
                family=selected['family']
                self.assertEqual({getattr(row,'run_id','NOT_A_PREDICTION_ROW') for row in rows},
                                 {family+'_left',family+'_right'})
                self.assertEqual((replicates,seed),(2000,7301))
                return {'status':'ok','critical':0.,'contrasts':{family:{'estimate':0.}}}
            api=self.api()
            with patch('research_v3.statistics.bootstrap_family',side_effect=bounded_bootstrap):
                out=api._statistics(api._Evidence(root),index,manifest,prediction_refs)
            self.assertEqual(len(out),9)

    def test_saved_rows_cannot_reinterpret_checkpoint_probability_scores_as_margins(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from research_v3.evaluate import _metrics,_read_predictions
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.prediction_fixture(root)
            rows=pq.read_table(path).to_pylist()
            for row in rows:row['score_kind']='svm_margin'
            pq.write_table(pa.Table.from_pylist(rows),path)
            receipt=read_json(root/'predictions/r.manifest.json')
            metrics=_metrics(_read_predictions(path),read_json(root/'data/prepared_manifest.json')['datasets']['yelp'])
            receipt.update(file_sha256=hash_file(path),metrics=metrics,metrics_hash=canonical_hash(metrics))
            save(root,'predictions/r.manifest.json',receipt)
            out=self.api().verify_research(root,'full_research')
            self.assertEqual(out['results'],[])
            self.assertTrue(any('score' in f['reason'].lower() for f in out['gates']['G3']['failures']))

    def test_authenticated_underpowered_confirmation_remains_reportable_but_blocks_full_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            api=self.api();e=api._Evidence(Path(tmp))
            for gate in e.gates.values():gate['checks']['other_synthetic_checks']={'status':'verified'}
            receipt={'status':'complete','n_actual':10,'n_planned':400,'underpowered':True,'mechanism_evidence_sufficient':False}
            e.check('G4','confirmation:synthetic','confirmation/receipt.json',lambda:receipt)
            fn=getattr(api,'_confirmation_sufficiency',None)
            if fn is None:self.fail('A separate confirmation sufficiency gate is missing')
            e.check('G4','confirmation_sufficiency:synthetic','confirmation/receipt.json',lambda:fn(receipt))
            self.assertTrue(e.gates['G4']['checks']['confirmation:synthetic'])
            self.assertEqual(e.gates['G4']['failures'][0]['status'],'pending')
            self.assertFalse(api._completion(e,True,'full_research')[0])

    def trajectory_fixture(self,root,only_svm=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from research_v3.trajectories import trace_failures
        from research_v3.types import PredictionRow
        conditions=[{'condition_id':'c1','dataset_id':'fixture','experiment_id':'E1','data_seed':2026,'metadata':{'size':1000}},
                    {'condition_id':'c6','dataset_id':'fixture','experiment_id':'E1','data_seed':2026,'metadata':{'size':6000}}]
        design={'conditions':conditions,'logical_runs':[],'physical_runs':[],'infeasible_conditions':[]}
        manifest={'runs':[]};paths={};refs={};result_refs=[]
        for model,seed in [('tfidf_svm',None),('distilbert',42)]:
            steps=[];predictions=[]
            for level in ('1','6'):
                rid=model+level;planned='planned-'+rid;cid='c'+level
                entry={'run_id':rid,'artifact_id':rid,'planned_run_id':planned,'condition_id':cid,'dataset_id':'fixture',
                       'model_id':model,'data_seed':2026,'training_seed':seed}
                manifest['runs'].append(entry);design['physical_runs'].append({'run_id':planned,'training_seed':seed})
                design['logical_runs'].append({'run_id':planned,'condition_id':cid,'dataset_id':'fixture','experiment_id':'E1','model_id':model,'context':{}})
                step={'run_id':rid,'condition_id':cid};steps.append(step)
                row=PredictionRow(rid,'x','base',(0,),(0,),(.8,.2),'probability','input');predictions.append(row)
                path=root/f'{rid}.parquet';pq.write_table(pa.Table.from_pylist([asdict(row)]),path)
                refs[rid]={'path':path.name,'sha256':hash_file(path)}
            if only_svm and model=='distilbert':continue
            spec={'model_id':model,'data_seed':2026,'training_seed':seed,'experiment_id':'E1','steps':steps}
            paths[model]=spec
            save(root,model+'.json',trace_failures(tuple(predictions),{model:spec},'single_label',2))
            result_refs.append({'dataset_id':'fixture','path_id':model,'path':model+'.json','sha256':hash_file(root/(model+'.json'))})
        save(root,'matrix/design_manifest.json',design);save(root,'run_manifest.json',manifest)
        save(root,'trajectory-paths.json',{'datasets':{'fixture':{'paths':paths}}})
        index={'trajectories':{'paths':{'path':'trajectory-paths.json','sha256':hash_file(root/'trajectory-paths.json')},'results':result_refs}}
        return index,{'datasets':{'fixture':{'task':'single_label','label_names':['a','b']}}},refs

    def test_empty_or_one_model_trajectory_declaration_cannot_cover_the_actual_matrix(self):
        for empty in (True,False):
            with self.subTest(empty=empty),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);index,prepared,refs=self.trajectory_fixture(root,only_svm=True)
                if empty:
                    save(root,'trajectory-paths.json',{'datasets':{}})
                    index['trajectories']['paths']['sha256']=hash_file(root/'trajectory-paths.json');index['trajectories']['results']=[]
                with self.assertRaisesRegex(PermissionError,'coverage|trajectory|representative'):
                    api=self.api();api._trajectory_summaries(api._Evidence(root),index,prepared,refs)

    def test_trajectory_model_seed_condition_and_order_are_bound_to_real_logical_runs(self):
        from research_v3.trajectories import trace_failures
        from research_v3.evaluate import _read_predictions
        for mutation in ('model','seed','condition','order'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);index,prepared,refs=self.trajectory_fixture(root)
                declared=read_json(root/'trajectory-paths.json');spec=declared['datasets']['fixture']['paths']['distilbert']
                if mutation=='model':spec['model_id']='tfidf_svm'
                elif mutation=='seed':spec['training_seed']=43
                elif mutation=='condition':spec['steps'][0]['condition_id']='invented'
                else:spec['steps'].reverse()
                rows=tuple(r for step in spec['steps'] for r in _read_predictions(root/refs[step['run_id']]['path']))
                save(root,'distilbert.json',trace_failures(rows,{'distilbert':spec},'single_label',2))
                save(root,'trajectory-paths.json',declared)
                index['trajectories']['paths']['sha256']=hash_file(root/'trajectory-paths.json')
                next(r for r in index['trajectories']['results'] if r['path_id']=='distilbert')['sha256']=hash_file(root/'distilbert.json')
                with self.assertRaisesRegex((PermissionError,ValueError),'trajectory|logical|representative|condition|seed|model|order'):
                    api=self.api();api._trajectory_summaries(api._Evidence(root),index,prepared,refs)

    def test_all_applicable_trajectory_groups_accept_genuine_representative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);index,prepared,refs=self.trajectory_fixture(root)
            api=self.api();out=api._trajectory_summaries(api._Evidence(root),index,prepared,refs)
            self.assertEqual(set(out['fixture']),{'tfidf_svm','distilbert'})

    def test_confirmation_cannot_use_a_different_authenticated_audit_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root,'audit/global.json',{'records':[{'sample_id':'actual-A'}]})
            index={'audit':{'frame':{'path':'audit/global.json','sha256':hash_file(root/'audit/global.json')}}}
            api=self.api();fn=getattr(api,'_confirmation_evidence',None)
            if fn is None:self.fail('Confirmation/global-A equivalence integration is missing')
            with patch('research_v3.confirmation.verify_confirmation_audit_frame',side_effect=ValueError('Confirmation excluded a different A frame')):
                with self.assertRaisesRegex(ValueError,'different A'):
                    fn(api._Evidence(root),index,{'pilot_id':'synthetic'},{})

    def claim_fixture(self,root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from research_v3.statistics import bootstrap_family
        from research_v3.types import PredictionRow
        predictions=[];refs={}
        for arm in ('left','right'):
            rows=[]
            for i in (0,1):
                predicted=i if arm=='left' else 1-i
                rows.append(PredictionRow(arm,'s'+str(i),'g'+str(i),(i,),(predicted,),
                    (.9,.1) if predicted==0 else (.1,.9),'probability','input'))
            path=root/(arm+'.parquet');pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]),path)
            refs[arm]={'path':path.name,'sha256':hash_file(path)};predictions.extend(rows)
        manifest={'datasets':{'yelp':{'task':'single_label','label_count':2,'partition':'Q'}},
            'runs':{r:{'dataset_id':'yelp','data_seed':2026,'training_seed':None,'evaluation_id':r} for r in ('left','right')},
            'estimates':[{'estimate_id':'delta','metric':'correctness','draws':[{'data_seed':2026,'left_runs':['left'],'right_runs':['right']}]}],
            'contrasts':[{'contrast_id':'c','family':'F6','dataset_id':'yelp','pilot_id':'synthetic',
                          'components':[{'estimate_id':'delta','coefficient':1}]}]}
        save(root,'contrasts.json',manifest)
        result=bootstrap_family(tuple(predictions),manifest,replicates=2000,seed=7301);save(root,'F6.json',result)
        stats=[{'family':'F6','contrasts':result['contrasts']}]
        cases=[{'sample_id':'s0','group_id':'g0'},{'sample_id':'s1','group_id':'g1'}]
        save(root,'claims/cases.json',{'dataset_id':'yelp','cases':cases})
        case_ref={'path':'claims/cases.json','sha256':hash_file(root/'claims/cases.json')}
        stat_ref={'family':'F6','path':'F6.json','sha256':hash_file(root/'F6.json'),
                  'contrast_file_sha256':hash_file(root/'contrasts.json')}
        claim={'claim_id':'effect','kind':'interventional','status':'supported','text':'Synthetic local paired response effect.',
               'dataset_id':'yelp','contrast_id':'c','direction':'positive','scope':'local_paired_intervention',
               'cases':case_ref,'evidence':[{'path':'F6.json','sha256':stat_ref['sha256']}]}
        case_hash=canonical_hash({'dataset_id':'yelp','cases':cases})
        binding={'contrast_file_sha256':hash_file(root/'contrasts.json'),'statistics_file_sha256':stat_ref['sha256'],
            'contrast':result['contrasts']['c'],'dataset_id':'yelp','contrast_id':'c','direction':'positive',
            'scope':'local_paired_intervention','cases_file_sha256':case_ref['sha256'],'case_set_hash':case_hash,
            'prediction_files':{r:refs[r]['sha256'] for r in sorted(refs)}}
        reviews=[{'claim_id':'effect','reviewer_id':person,'decision':'supported','rationale':'Synthetic reviewed local scope and direction.',
            'claim_content_hash':canonical_hash(claim),'evidence_hash':canonical_hash(binding),'dataset_id':'yelp','contrast_id':'c',
            'direction':'positive','scope':'local_paired_intervention','case_set_hash':case_hash,'case_count':2}
            for person in ('synthetic_A','synthetic_B')]
        save(root,'claims/reviews.json',{'records':reviews})
        index={'statistics':[stat_ref],'claim_reviews':{'path':'claims/reviews.json','sha256':hash_file(root/'claims/reviews.json')}}
        confirmations=[{'pilot_id':'synthetic','status':'complete','underpowered':False,'mechanism_evidence_sufficient':True,
            'n_actual':2,'n_planned':2,
            'views':{r:{'dataset_id':'yelp','evaluation_id':r,'file':refs[r]['path'],'sha256':refs[r]['sha256']} for r in refs}}]
        audit={'reviewed_sample_ids':['yelp:A-case-outside-confirmation']}
        return index,claim,manifest,stats,refs,audit,confirmations

    def test_supported_local_claim_requires_exact_case_effect_and_two_bound_human_interpretations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.claim_fixture(root);api=self.api()
            fn=getattr(api,'_verify_interventional_claim',None)
            if fn is None:self.fail('Exact reviewed interventional-claim verification is missing')
            out=fn(api._Evidence(root),*args)
            self.assertEqual(out['status'],'supported')
            self.assertEqual(out['dataset_id'],'yelp')

    def test_claim_cannot_borrow_another_dataset_cases_direction_or_unbound_reviewer_receipt(self):
        for mutation in ('dataset','cases','interval','direction','one_reviewer','review_hash','missing_reviews','scope'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);index,claim,manifest,stats,refs,audit,confirmations=self.claim_fixture(root)
                if mutation=='dataset':claim['dataset_id']='banking77'
                elif mutation=='cases':
                    save(root,'claims/cases.json',{'dataset_id':'yelp','cases':[{'sample_id':'unrelated-A','group_id':'other'}]})
                    claim['cases']['sha256']=hash_file(root/'claims/cases.json')
                elif mutation=='interval':
                    stats[0]['contrasts']['c']['simultaneous_lower']=-.1
                    saved=read_json(root/'F6.json');saved['contrasts']['c']=stats[0]['contrasts']['c'];save(root,'F6.json',saved)
                    index['statistics'][0]['sha256']=hash_file(root/'F6.json')
                elif mutation=='direction':claim['direction']='negative'
                elif mutation=='scope':claim['scope']='universal_model_ability'
                elif mutation=='missing_reviews':index.pop('claim_reviews')
                else:
                    reviews=read_json(root/'claims/reviews.json')
                    if mutation=='one_reviewer':reviews['records'][1]['reviewer_id']=reviews['records'][0]['reviewer_id']
                    else:reviews['records'][1]['evidence_hash']='not-the-actual-evidence'
                    save(root,'claims/reviews.json',reviews);index['claim_reviews']['sha256']=hash_file(root/'claims/reviews.json')
                api=self.api();fn=getattr(api,'_verify_interventional_claim',None)
                if fn is None:self.fail('Exact reviewed interventional-claim verification is missing')
                with self.assertRaisesRegex((PermissionError,ValueError),'claim|dataset|case|direction|interval|review|scope|evidence'):
                    fn(api._Evidence(root),index,claim,manifest,stats,refs,audit,confirmations)

    def test_trajectory_ledger_preserves_logical_aliases_full_draw_and_design_infeasible_exceptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.trajectory_fixture(root)
            design=read_json(root/'matrix/design_manifest.json');manifest=read_json(root/'run_manifest.json')
            def add(experiment,cid,model,metadata,context=None,data_seed=2026):
                if not any(c['condition_id']==cid for c in design['conditions']):
                    design['conditions'].append({'condition_id':cid,'dataset_id':'fixture','experiment_id':experiment,'data_seed':data_seed,'metadata':metadata})
                rid=model+'-'+cid;planned='planned-'+rid
                design['physical_runs'].append({'run_id':planned,'training_seed':42})
                manifest['runs'].append({'run_id':rid,'artifact_id':rid,'planned_run_id':planned,'condition_id':cid,
                    'dataset_id':'fixture','model_id':model,'data_seed':data_seed,'training_seed':42})
                design['logical_runs'].append({'run_id':planned,'condition_id':cid,'dataset_id':'fixture','experiment_id':experiment,
                    'model_id':model,'context':context or {}})
                return planned
            add('E4','minus','distilbert',{'size':6000,'alpha':-.5},{'alpha':-.5})
            add('E4','plus','distilbert',{'size':6000,'alpha':.5},{'alpha':.5})
            design['logical_runs'].append({'run_id':'planned-distilbert6','condition_id':'c6','dataset_id':'fixture',
                'experiment_id':'E4','model_id':'distilbert','context':{'alpha':0}})
            add('E1','full','distilbert',{'size':'full','full_pool':True},data_seed=0)
            for rho in (1,10,50):
                planned=add('E3','rho'+str(rho),'distilbert',{'requested_ratio':rho},
                            {'ratio':rho,'head_seed':None if rho==1 else 9101})
                if rho!=1:
                    design['logical_runs'].append({'run_id':planned,'condition_id':'rho'+str(rho),'dataset_id':'fixture',
                        'experiment_id':'E3','model_id':'distilbert','context':{'ratio':rho,'head_seed':9102}})
            design['logical_runs'].append({'run_id':'planned-distilbert6','condition_id':'c6','dataset_id':'fixture',
                'experiment_id':'E5','model_id':'distilbert','context':{}})
            add('E5','c100','distilbert',{'size':100000})
            add('E5','c6','modernbert',{'size':6000})
            add('E5','c100','modernbert',{'size':100000})
            design['infeasible_conditions']=[{'dataset_id':'fixture','experiment_id':'E2','reason':'Synthetic declared no feasible shot capacity.'}]
            catalogue=self.api()._trajectory_catalogue(design,manifest)
            self.assertIn(('fixture','E2','tfidf_svm'),catalogue)
            self.assertFalse(catalogue[('fixture','E2','tfidf_svm')]['required'])
            self.assertEqual(catalogue[('fixture','E2','tfidf_svm')]['reason'],'all_conditions_declared_infeasible')
            self.assertEqual([r['level'] for r in catalogue[('fixture','E4','distilbert')]['nodes']],[-.5,0,.5])
            self.assertEqual([r['level'] for r in catalogue[('fixture','E3','distilbert')]['nodes']],[1,10,50])
            self.assertEqual(catalogue[('fixture','E3','distilbert')]['head_seed'],9101)
            self.assertEqual(catalogue[('fixture','E1','distilbert')]['nodes'][-1]['data_seed'],0)
            self.assertTrue(catalogue[('fixture','E5','modernbert')]['required'])
            self.assertTrue(catalogue[('fixture','E5','distilbert')]['required'])


if __name__=='__main__':unittest.main()
