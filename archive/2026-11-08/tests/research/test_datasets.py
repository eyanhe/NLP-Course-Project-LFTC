"""Catch label collapse, split-ID collisions and lost question/answer fields."""
import importlib
import unittest


class DatasetTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('research_v3.datasets')
        except ModuleNotFoundError:
            self.fail('research_v3.datasets is not implemented')

    def test_goemotions_keeps_neutral_cooccurrence(self):
        rows = [{'id': 'x', 'text': 'A mixed reaction', 'labels': [6, 9, 27]}]
        result = tuple(self.api().examples_from_rows('go_emotions', 'train', rows, tuple(str(i) for i in range(28))))
        self.assertEqual(result[0].labels, (6, 9, 27))

    def test_banking_string_and_numeric_labels_use_declared_order(self):
        names = ('pending', 'failed')
        rows = [{'text': 'not completed', 'label': 'failed'}, {'text': 'not finished', 'label': 1}]
        result = tuple(self.api().examples_from_rows('banking77', 'train', rows, names))
        self.assertEqual([r.labels for r in result], [(1,), (1,)])

    def test_yahoo_keeps_title_and_answer_when_question_is_empty(self):
        row = {'id': 7, 'topic': 0, 'question_title': 'A title', 'question_content': '', 'best_answer': 'An answer'}
        result = tuple(self.api().examples_from_rows('yahoo_answers_topics', 'test', [row], ('topic',)))
        self.assertEqual(result[0].text, '[TITLE] A title [QUESTION]  [ANSWER] An answer')
        self.assertEqual(result[0].sample_id, 'yahoo_answers_topics/test/7')

    def test_unknown_label_fails_before_preparation(self):
        with self.assertRaises(ValueError):
            tuple(self.api().examples_from_rows('banking77', 'train', [{'text': 'x', 'label': 9}], ('only',)))


if __name__ == '__main__':
    unittest.main()
