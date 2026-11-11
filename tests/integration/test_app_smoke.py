"""PPT narrative pages and the original saved-model demo."""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ['Overview', 'Performance & cost', 'Robustness', 'Mechanisms',
         'Live prediction', 'Methods & evidence', 'Conclusions']


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'optional Streamlit is not installed')
class AppSmokeTests(unittest.TestCase):
    def _start(self, root):
        from streamlit.testing.v1 import AppTest
        old = os.environ.get('LFTC_ROOT')
        os.environ['LFTC_ROOT'] = str(root)
        self.addCleanup(lambda: os.environ.pop('LFTC_ROOT', None) if old is None
                        else os.environ.__setitem__('LFTC_ROOT', old))
        return AppTest.from_file(str(ROOT / 'apps/streamlit_app.py'), default_timeout=45).run()

    def test_ppt_story_and_all_seven_pages_render(self):
        app = self._start(ROOT)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(app.sidebar.radio[0].options, PAGES)
        self.assertTrue(any('When does' in item.value and 'pretraining' in item.value
                            for item in app.markdown))
        for page in PAGES[1:]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(app.exception), 0, f'{page}: {app.exception}')

    def test_robustness_and_mechanism_controls_render_actual_studies(self):
        app = self._start(ROOT)
        app.sidebar.radio[0].set_value('Robustness').run()
        for study in ['Sentence pairs', 'Cross-dataset transfer', 'Class imbalance']:
            app.radio[0].set_value(study).run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
        app.sidebar.radio[0].set_value('Mechanisms').run()
        for study in ['Label support', 'Word swaps', 'Optimization', 'Pretraining & adaptation']:
            app.radio[0].set_value(study).run()
            self.assertEqual(len(app.exception), 0, str(app.exception))

    def test_fresh_checkout_retains_story_and_explains_missing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copyfile(ROOT / 'pyproject.toml', root / 'pyproject.toml')
            shutil.copytree(ROOT / 'configs', root / 'configs')
            app = self._start(root)
            self.assertEqual(len(app.exception), 0, str(app.exception))
            for page in PAGES[1:]:
                app.sidebar.radio[0].set_value(page).run()
                self.assertEqual(len(app.exception), 0, f'{page}: {app.exception}')

    def test_demo_uses_existing_benchmark_model_and_pair_inputs(self):
        app = self._start(ROOT)
        app.sidebar.radio[0].set_value('Live prediction').run()
        self.assertEqual(len(app.multiselect[0].options), 5)
        app.multiselect[0].set_value(['logistic_regression'])
        app.text_area[0].set_value('A beautifully acted film with a moving ending.')
        next(button for button in app.button if button.label == 'Compare models').click().run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        result = app.session_state['demo_result']
        self.assertEqual([row['model'] for row in result['rows']], ['logistic_regression'])
        self.assertIn(result['rows'][0]['prediction'], ['Negative', 'Positive'])
        app.selectbox[0].set_value('mnli').run()
        self.assertEqual(len(app.text_area), 2)
        self.assertEqual([area.label for area in app.text_area], ['Premise', 'Hypothesis'])
