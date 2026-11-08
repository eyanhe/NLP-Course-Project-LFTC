"""Tokenizer cost evidence must describe actual model tokens on the fixed text."""
from dataclasses import asdict
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest

from research_v3.storage import read_json
from research_v3.types import RunSpec,VisibleInput


class CostTests(unittest.TestCase):
    def api(self):
        try:return importlib.import_module('research_v3.costs')
        except ModuleNotFoundError:self.fail('own tokenizer cost recording is missing')

    def fixture(self,root):
        from transformers import BertTokenizerFast
        directory=root/'tokenizer';directory.mkdir()
        (directory/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nun\n##happy\ngood\n',encoding='utf-8')
        tok=BertTokenizerFast(vocab_file=str(directory/'vocab.txt'));tok.save_pretrained(directory)
        def row(sid,text):return VisibleInput(sid,text,0,len(text),99,99,hashlib.sha256(text.encode()).hexdigest())
        return tok,(row('T/0','unhappy'),),(row('V/0','good'),),directory

    def test_model_own_lengths_do_not_reuse_common_token_metadata(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            tokenizer,train,_,_=self.fixture(Path(temp))
            out=api.own_token_counts(tokenizer,train)
            self.assertEqual(out['content_tokens'],2)
            self.assertEqual(out['tokens_with_specials'],4)
            self.assertEqual(out['rows'],1)
            self.assertFalse(out['includes_padding'])

    def test_budget_is_bound_to_exact_authorized_role_ids_and_tokenizer(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,train,validation,directory=self.fixture(root)
            ih=lambda ids:hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
            run=RunSpec('run','condition','modernbert',42,{'role_ids_hash':{'train':ih(['T/0']),
                'select':ih(['V/0'])}},'input','assets','code',False)
            assets={'asset_hash':'assets','models':{'modernbert':{'directory':str(directory),'revision':'a'*40}}}
            out=api.record_token_budget(run,train,validation,assets,root)
            self.assertEqual(out['roles']['train']['content_tokens'],2)
            self.assertEqual(out['roles']['select']['tokens_with_specials'],3)
            self.assertEqual(out,api.record_token_budget(run,train,validation,assets,root))
            self.assertEqual(read_json(root/'training/token-budgets/run.json'),out)
            with self.assertRaisesRegex(ValueError,'role|IDs'):
                api.record_token_budget(run,validation,train,assets,root)


if __name__=='__main__':unittest.main()
