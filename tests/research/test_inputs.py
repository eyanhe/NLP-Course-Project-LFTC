"""Catch mismatched token/character budgets and lossy normalization."""
import importlib
from pathlib import Path
import unittest

from research_v3.types import Example

ROOT = Path(__file__).resolve().parents[2]


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'assets/distilbert', local_files_only=True, use_fast=True)

    def api(self):
        try:
            return importlib.import_module('research_v3.inputs')
        except ModuleNotFoundError:
            self.fail('research_v3.inputs is not implemented')

    def test_budget_is_254_content_tokens_with_exact_character_offsets(self):
        row = Example('imdb', 'x', ' '.join(['good'] * 300), (1,), 'train')
        result = self.api().make_visible_input(row, self.tokenizer)
        self.assertEqual(result.content_tokens, 254)
        self.assertEqual(result.full_tokens, 300)
        self.assertEqual(result.text.count('good'), 254)
        self.assertEqual(result.end_char, 1269)
        self.assertEqual(len(self.tokenizer(result.text)['input_ids']), 256)

    def test_negation_punctuation_and_emoji_survive_cleaning(self):
        row = Example('imdb', 'x', '<br />This is NOT good! 😟', (0,), 'test')
        result = self.api().make_visible_input(row, self.tokenizer)
        self.assertEqual(result.text, 'This is NOT good! 😟')

    def test_unseen_tail_does_not_change_the_common_visible_input(self):
        prefix = ' '.join(['good'] * 300)
        a = Example('imdb', 'a', prefix + ' ending A', (1,), 'test')
        b = Example('imdb', 'b', prefix + ' radically different ending B', (0,), 'test')
        one = self.api().make_visible_input(a, self.tokenizer)
        two = self.api().make_visible_input(b, self.tokenizer)
        self.assertEqual(one.text, two.text)
        self.assertEqual(one.text_hash, two.text_hash)

    def test_unicode_character_span_round_trips(self):
        row = Example('go_emotions', 'u', 'Straße café 😟 not happy', (2,), 'train')
        result = self.api().make_visible_input(row, self.tokenizer, limit=3)
        self.assertEqual(result.text, row.text[result.start_char:result.end_char])
        self.assertLessEqual(result.content_tokens, 3)


if __name__ == '__main__':
    unittest.main()
