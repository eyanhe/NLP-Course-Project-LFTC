"""Forward-only preparation cannot train, produce gradients or alter weights."""
import importlib
import unittest


class ForwardTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.smoke')
        except ModuleNotFoundError:
            self.fail('forward-only model checks are missing')

    def test_multilabel_forward_preserves_head_and_creates_no_gradients(self):
        api=self.api()
        import torch
        from transformers import DistilBertConfig,DistilBertForSequenceClassification
        config=DistilBertConfig(vocab_size=30,n_layers=1,n_heads=2,dim=8,hidden_dim=16,
                               num_labels=28,problem_type='multi_label_classification')
        model=DistilBertForSequenceClassification(config)
        before=model.classifier.weight.detach().clone()
        batch={'input_ids':torch.tensor([[1,2,3,4],[1,5,6,0]]),
               'attention_mask':torch.tensor([[1,1,1,1],[1,1,1,0]])}
        result=api.check_forward(model,batch,28,'multi_label')
        self.assertEqual(result['logits_shape'],[2,28])
        self.assertFalse(result['gradients_created'])
        self.assertTrue(torch.equal(model.classifier.weight,before))
        self.assertFalse(result['actual_model_training_run'])

    def test_wrong_label_dimension_is_rejected(self):
        api=self.api()
        import torch
        from transformers import DistilBertConfig,DistilBertForSequenceClassification
        model=DistilBertForSequenceClassification(DistilBertConfig(vocab_size=30,n_layers=1,
              n_heads=2,dim=8,hidden_dim=16,num_labels=2))
        with self.assertRaisesRegex(ValueError,'shape|label'):
            api.check_forward(model,{'input_ids':torch.tensor([[1,2]])},28,'multi_label')

    def test_synthetic_batch_reaches_configured_microbatch_and_token_budget(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('assets/distilbert',local_files_only=True)
        api=self.api()
        self.assertTrue(hasattr(api,'synthetic_visible_texts'),'microbatch boundary packet is missing')
        texts=api.synthetic_visible_texts(tokenizer,4)
        batch=tokenizer(texts,padding=True,return_tensors='pt')
        self.assertEqual(list(batch['input_ids'].shape),[4,256])


if __name__=='__main__':
    unittest.main()
