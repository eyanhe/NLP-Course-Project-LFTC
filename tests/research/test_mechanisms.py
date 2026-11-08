"""Mechanism fits must change the intended factor and preserve the others."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import research_v3  # Arrow must load before Torch on Windows.
import numpy as np
import torch
from transformers import BertTokenizerFast, DistilBertConfig

from research_v3.features import WordFeatures, fit_features
from research_v3.types import Example, PreparedDataset, RunSpec, VisibleInput


class MechanismTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.diagnostics')
        except ModuleNotFoundError:
            self.fail('mechanism fitting support is missing')

    def config(self):
        return DistilBertConfig(vocab_size=12, max_position_embeddings=32, n_layers=1,
            n_heads=2, dim=8, hidden_dim=12, num_labels=2, dropout=0.,
            attention_dropout=0., seq_classif_dropout=0.)

    def fixture(self, root, model='m2_distilbert', strict=False, **hp):
        texts=['bad red', 'good blue', 'red bad', 'blue good']
        rows=tuple(Example('tiny',f'train/{i}',text,(i%2,),'train') for i,text in enumerate(texts))
        val=tuple(Example('tiny',f'select/{i}',text,(i%2,),'validation') for i,text in enumerate(texts[:2]))
        inputs={r.sample_id:VisibleInput(r.sample_id,r.text,0,len(r.text),2,2,r.sample_id)
                for r in (*rows,*val)}
        class Strict(dict):
            def __getitem__(self, key):
                if key in {'select','discovery','test'}:
                    raise AssertionError('forbidden holdout access')
                return super().__getitem__(key)
        parts={'train':rows,'select':val,'discovery':(),'test':()}
        data=PreparedDataset('tiny','single_label',('bad','good'),Strict(parts) if strict else parts,'pool')
        (root/'training/conditions').mkdir(parents=True)
        (root/'training/conditions/supplied-condition.json').write_text(json.dumps({
            'dataset_id':'tiny','train_ids':[r.sample_id for r in rows],
            'metadata':{'pool_manifest_hash':'pool'}}),encoding='utf-8')
        source=root/'tokenizer';source.mkdir()
        (source/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nbad\nred\ngood\nblue\nx\ny\nz\n',encoding='utf-8')
        tokenizer=BertTokenizerFast(vocab_file=str(source/'vocab.txt'),do_lower_case=True)
        tokenizer.save_pretrained(source)
        assets={'models':{'distilbert':{'directory':str(source),'revision':'a'*40}}}
        run=RunSpec('supplied-run','supplied-condition',model,42,{
            'device':'cpu','cpu_threads':1,'arm':'random_unfrozen','epochs':10,
            'effective_batch':16,'micro_batch':4,**hp},'input','assets','main-code',strict)
        return data,inputs,assets,run,tokenizer

    # Removing explicit head copying or freezing the head should break this.
    def test_m2_four_arms_share_head_and_only_encoder_gradient_switches(self):
        api=self.api();cfg=self.config()
        pretrained=api.build_m2_model(cfg,'random_unfrozen',17).distilbert.state_dict()
        arms=[api.build_m2_model(cfg,arm,42,pretrained_encoder_state=pretrained)
              for arm in ['pretrained_frozen','pretrained_unfrozen','random_frozen','random_unfrozen']]
        for name,value in pretrained.items():
            torch.testing.assert_close(arms[0].distilbert.state_dict()[name],value,rtol=0,atol=0)
        self.assertFalse(torch.equal(arms[0].distilbert.embeddings.word_embeddings.weight,
                                     arms[2].distilbert.embeddings.word_embeddings.weight))
        x={'input_ids':torch.tensor([[2,5,6,3]]),'attention_mask':torch.ones((1,4),dtype=torch.long)}
        expected={n:p.detach().clone() for n,p in arms[0].named_parameters() if not n.startswith('distilbert.')}
        for index,model in enumerate(arms):
            for name,value in expected.items():
                torch.testing.assert_close(dict(model.named_parameters())[name],value,rtol=0,atol=0)
            model(**x).logits.sum().backward()
            self.assertTrue(all(p.grad is not None for n,p in model.named_parameters() if not n.startswith('distilbert.')))
            self.assertEqual(any(p.grad is not None for p in model.distilbert.parameters()), index%2==1)
        groups=api.m2_optimizer_groups(arms[1])
        self.assertEqual([g['lr'] for g in groups],[.001,.00002])
        self.assertEqual({id(p) for g in groups for p in g['params']},
                         {id(p) for p in arms[1].parameters() if p.requires_grad})

    # Coupling head or batch streams to model construction consumes these values.
    def test_model_construction_preserves_global_rng_and_batch_orders(self):
        api=self.api();torch.random.default_generator.manual_seed(99);state=torch.random.get_rng_state().clone()
        a=api.build_m2_model(self.config(),'random_unfrozen',42)
        torch.testing.assert_close(torch.random.get_rng_state(),state,rtol=0,atol=0)
        order=api.epoch_order(7,42,1)
        api.build_m2_model(self.config(),'random_frozen',42)
        np.testing.assert_array_equal(api.epoch_order(7,42,1),order)
        b=api.build_m2_model(self.config(),'random_unfrozen',43)
        self.assertFalse(torch.equal(a.classifier.weight,b.classifier.weight))

    # Misusing weighted CE's sum-of-weights normalization changes loss/gradient.
    def test_m3_weights_preserve_n0_and_weighted_risk_gradient(self):
        api=self.api();y=np.array([0,0,1]);q=np.array([.25,.75])
        weights=api.m3_sample_weights(y,q,12)
        np.testing.assert_allclose(weights,[1.5,1.5,9.],rtol=0,atol=1e-12)
        z=torch.zeros((3,2),requires_grad=True)
        loss=api.m3_head_loss(z,torch.tensor(y),torch.tensor([.375,.375,2.25]))
        self.assertAlmostEqual(loss.item(),np.log(2),places=6)
        loss.backward()
        np.testing.assert_allclose(z.grad.numpy(),[[-.0625,.0625],[-.0625,.0625],[.375,-.375]],rtol=0,atol=1e-7)

    # Tail reductions must delete within U0, follow semantic order and retain classes.
    def test_tail_reduction_tracks_head_order_and_never_adds_labeled_rows(self):
        api=self.api();ids=tuple(f'{c}/{i}' for c in range(3) for i in range(4));y=np.repeat([0,1,2],4)
        selected=api.tail_reduced_indices(ids,y,(0,1,2),10,2026)
        self.assertEqual(np.bincount(y[selected]).tolist(),[4,4,1])
        other=api.tail_reduced_indices(ids,y,(2,1,0),10,2026)
        self.assertEqual(np.bincount(y[other]).tolist(),[1,4,4])
        self.assertEqual(api.current_tail_labels((1,0)),(0,))
        self.assertEqual(api.natural_tail_labels([4,1,1]),(1,2))
        self.assertIsNone(api.natural_tail_labels([4,4,4]))
        for labels,q,n0 in [([0,0],[.5,.5],2),([0,1],[.5,.5],1)]:
            with self.assertRaises(ValueError):api.m3_sample_weights(np.array(labels),np.array(q),n0)

    # New vocabulary must not alter shared rows, encoder/head initialization or cache.
    def test_m5_paired_initialization_copies_rows_and_aligns_nonembedding_state(self):
        api=self.api()
        base=WordFeatures({'red':2,'blue':3},np.arange(16,dtype=np.float32).reshape(4,4),False,('a',),{})
        large=WordFeatures({'red':2,'blue':3,'extra':4},np.full((5,4),-7,dtype=np.float32),False,('a',),{})
        before=large.embedding_matrix.copy()
        paired=api.paired_word_features(base,large,42)
        np.testing.assert_array_equal(paired.embedding_matrix[:4],base.embedding_matrix)
        np.testing.assert_array_equal(large.embedding_matrix,before)
        np.testing.assert_array_equal(api.paired_word_features(base,large,42).embedding_matrix,paired.embedding_matrix)
        a=api.build_paired_bilstm(base,base,42,2,{'hidden_per_direction':3})
        b=api.build_paired_bilstm(base,paired,42,2,{'hidden_per_direction':3})
        for name,value in a.state_dict().items():
            torch.testing.assert_close(b.state_dict()[name][:4] if name=='embedding.weight' else b.state_dict()[name],value,rtol=0,atol=0)

    # A complete status needs a real original-before-save score comparison and hashes.
    def test_m2_fit_trains_all_ten_epochs_and_reload_hashes_are_enforced(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root)
            out=root/'runs'/run.run_id
            result=api.fit_mechanism_run(run,data,inputs,assets,out,model_config=self.config(),tokenizer=tok)
            self.assertEqual(result['status'],'complete');self.assertTrue(result['reload_verified'])
            self.assertEqual(result['epochs_completed'],10);self.assertEqual(result['optimizer_updates'],10)
            self.assertEqual(len(json.loads((out/'history.json').read_text())['epochs']),10)
            self.assertEqual(result['run']['condition_id'],'supplied-condition')
            self.assertEqual(result['run']['code_hash'],'main-code')
            self.assertIn('diagnostics_code_hash',result)
            self.assertEqual(result['training_unique_rows'],4)
            self.assertEqual(result['holdout_labels_accessed'],2)
            self.assertTrue(api.fit_mechanism_run(run,data,inputs,assets,out,model_config=self.config(),tokenizer=tok)['reused_complete'])
            with self.assertRaisesRegex(ValueError,'identity|request'):
                api.fit_mechanism_run(replace(run,training_seed=43),data,inputs,assets,out,model_config=self.config(),tokenizer=tok)
            changed=dict(inputs);changed['train/0']=replace(inputs['train/0'],text='good blue')
            with self.assertRaisesRegex(ValueError,'identity|request'):
                api.fit_mechanism_run(run,data,changed,assets,out,model_config=self.config(),tokenizer=tok)
            (out/'checkpoint/model.safetensors').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'hash|changed'):
                api.score_mechanism_saved(out,tuple(inputs.values())[:2])

    # Fixed representation is supplied from full T even when labeled support is reduced.
    def test_m3_svm_uses_supplied_full_t_features_and_weights_then_reloads(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root,model='fixed_tfidf_svm',
                target_q=[.25,.75],reference_n=8,unlabeled_pool_manifest='pool')
            full=tuple(inputs[r.sample_id] for r in data.partitions['train'])
            features=fit_features('tfidf_svm',full,{'ngram_range':[1,1]})
            out=root/'runs'/run.run_id
            result=api.fit_mechanism_run(run,data,inputs,assets,out,shared_features=features)
            self.assertEqual(result['sample_weight_sum'],8.)
            self.assertEqual(result['representation_unlabeled_pool_manifest'],'pool')
            self.assertTrue(result['reload_verified'])
            scores=api.score_mechanism_saved(out,full)
            self.assertEqual(scores.shape,(4,2));self.assertTrue(np.isfinite(scores).all())
            from sklearn.svm import LinearSVC
            independently_fitted=LinearSVC(C=1,tol=.0001,max_iter=10000,dual='auto',random_state=2026)
            independently_fitted.fit(features.transform(full),[0,1,0,1],sample_weight=[1,3,1,3])
            margin=independently_fitted.decision_function(features.transform(full))
            np.testing.assert_allclose(scores,np.column_stack([-margin,margin]),rtol=1e-12,atol=1e-12)

    # Sampling without replacement, update-by-support-size and encoder drift are bugs.
    def test_m3_frozen_head_fixed_updates_exposures_and_checkpoint_encoder(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root,model='frozen_generic_distilbert_head',
                target_q=[.25,.75],reference_n=8,unlabeled_pool_manifest='pool')
            pretrained=api.build_m2_model(self.config(),'random_unfrozen',17).distilbert.state_dict()
            out=root/'runs'/run.run_id
            result=api.fit_mechanism_run(run,data,inputs,assets,out,model_config=self.config(),tokenizer=tok,
                pretrained_encoder_state=pretrained)
            self.assertEqual(result['optimizer_updates'],5)
            self.assertEqual(result['training_exposures'],80)
            self.assertEqual(sum(result['exposure_by_sample_id'].values()),80)
            self.assertEqual(result['cls_cache_rows_encoded'],6)
            self.assertTrue(result['reload_verified'])
            from transformers import DistilBertForSequenceClassification
            saved=DistilBertForSequenceClassification.from_pretrained(out/'checkpoint',local_files_only=True)
            for name,value in pretrained.items():
                torch.testing.assert_close(saved.distilbert.state_dict()[name],value,rtol=0,atol=0)

    # Strict tasks cannot leak V/D or use a full-T representation to impersonate E2.
    def test_strict_mechanism_rejected_before_reading_holdouts(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root,strict=True)
            with self.assertRaisesRegex(ValueError,'strict|E2'):
                api.fit_mechanism_run(run,data,inputs,assets,root/'runs'/run.run_id,
                    model_config=self.config(),tokenizer=tok)

    # M5 must actually execute the paired model and all requested epochs.
    def test_m5_bilstm_fit_saves_compatible_bundle_and_ten_epoch_trajectory(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root,model='bilstm',
                hidden_per_direction=2,dropout=0.,lr=.001,effective_batch=64,max_epochs=10)
            features=WordFeatures({'bad':2,'red':3,'good':4,'blue':5},np.ones((6,4),np.float32),False,
                                 tuple(r.sample_id for r in data.partitions['train']),{})
            result=api.fit_mechanism_run(run,data,inputs,assets,root/'runs'/run.run_id,
                shared_features=features,baseline_features=features)
            self.assertTrue(result['reload_verified']);self.assertEqual(result['epochs_completed'],10)
            self.assertEqual(result['optimizer_updates'],10)

    # The baseline alias is honest only when the optimizer/dropout trajectory is identical.
    def test_m5_baseline_training_matches_existing_core_given_same_features(self):
        api=self.api()
        from research_v3.train import _fit_neural, _neural_scores
        from research_v3.models import build_model
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root,model='bilstm',
                hidden_per_direction=2,dropout=.3,lr=.001,effective_batch=64,max_epochs=10)
            matrix=np.arange(24,dtype=np.float32).reshape(6,4)/20
            features=WordFeatures({'bad':2,'red':3,'good':4,'blue':5},matrix,False,
                                 tuple(r.sample_id for r in data.partitions['train']),{})
            train=tuple(inputs[r.sample_id] for r in data.partitions['train'])
            val=tuple(inputs[r.sample_id] for r in data.partitions['select'])
            core=root/'core';core.mkdir()
            original=_fit_neural(run,data,train,np.array([0,1,0,1]),val,np.array([0,1]),features,assets,core)
            out=root/'runs'/run.run_id
            paired=api.fit_mechanism_run(run,data,inputs,assets,out,shared_features=features,baseline_features=features)
            self.assertEqual(paired['selected_epoch'],original['selected_epoch'])
            a=load_file(str(core/'checkpoint/model.safetensors'))
            b=load_file(str(out/'checkpoint/model.safetensors'))
            for name,value in a.items():torch.testing.assert_close(b[name],value,rtol=0,atol=0)

    # Caching the mean token state, reusing a changed encoder, or silently accepting corruption breaks this.
    def test_generic_cls_cache_matches_real_cls_and_reuses_only_identical_encoder(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root)
            rows=tuple(inputs[r.sample_id] for r in data.partitions['train'])
            model=api.build_m2_model(self.config(),'random_frozen',42)
            cache=api.GenericCLSCache(root/'cache')
            values=cache.encode(model,rows,tok,'cpu',2)
            encoded=tok([r.text for r in rows],padding=True,return_tensors='pt')
            batch={k:v for k,v in encoded.items() if k in {'input_ids','attention_mask'}}
            model.eval()
            with torch.inference_mode():
                states=model.distilbert(**batch).last_hidden_state
                expected=states[:,0,:].numpy()
                actual_logits=model.classifier(model.dropout(torch.relu(model.pre_classifier(torch.tensor(values)))))
                torch.testing.assert_close(actual_logits,model(**batch).logits,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(values,expected,rtol=1e-6,atol=1e-7)
            self.assertFalse(np.allclose(values,states.mean(dim=1).numpy()))
            second=api.GenericCLSCache(root/'cache');second.encode(model,rows,tok,'cpu',2)
            self.assertEqual(second.rows_encoded,0)
            with torch.no_grad():model.distilbert.embeddings.word_embeddings.weight[5].add_(1.)
            second.encode(model,rows,tok,'cpu',2);self.assertEqual(second.rows_encoded,4)
            path=next((root/'cache').glob('*.npz'))
            with path.open('wb') as stream:np.savez(stream,cls=np.zeros(8,np.float32),sha256='bad')
            with self.assertRaisesRegex(ValueError,'hash'):
                api.GenericCLSCache(root/'cache').encode(api.build_m2_model(self.config(),'random_frozen',42),rows,tok,'cpu',2)

    # Distinct multilabel masks and constant columns must remain scoreable across all10 epochs.
    def test_m2_pretrained_frozen_multilabel_retains_constant_columns(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,inputs,assets,run,tok=self.fixture(root,arm='pretrained_frozen')
            data=replace(data,task='multi_label',label_names=('bad','good','constant'),partitions={
                **data.partitions,'train':tuple(replace(r,labels=(*r.labels,2)) for r in data.partitions['train']),
                'select':tuple(replace(r,labels=(*r.labels,2)) for r in data.partitions['select'])})
            state=api.build_m2_model(self.config(),'random_unfrozen',17).distilbert.state_dict()
            out=root/'runs'/run.run_id
            result=api.fit_mechanism_run(run,data,inputs,assets,out,model_config=self.config(),tokenizer=tok,
                                        pretrained_encoder_state=state)
            self.assertEqual(result['epochs_completed'],10)
            scores=api.score_mechanism_saved(out,tuple(inputs[r.sample_id] for r in data.partitions['train']))
            np.testing.assert_array_equal(scores[:,2],np.full(4,30.))
            from transformers import DistilBertForSequenceClassification
            model=DistilBertForSequenceClassification.from_pretrained(out/'checkpoint',local_files_only=True)
            for name,value in state.items():torch.testing.assert_close(model.distilbert.state_dict()[name],value,rtol=0,atol=0)

    # Declared sensitivity rates must enter the actual encoder parameter group.
    def test_m2_all_declared_sensitivity_rates_are_used_in_optimizer(self):
        api=self.api();model=api.build_m2_model(self.config(),'random_unfrozen',42)
        for rate in [1e-5,5e-5,5e-4]:
            groups=api.m2_optimizer_groups(model,rate)
            optimizer=torch.optim.AdamW(groups,weight_decay=.01)
            self.assertEqual([g['lr'] for g in optimizer.param_groups],[.001,rate])
            self.assertTrue(all(g['weight_decay']==.01 for g in optimizer.param_groups))


if __name__=='__main__':unittest.main()
