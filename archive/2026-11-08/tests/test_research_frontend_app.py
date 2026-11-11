"""Exercise actual Streamlit widgets against the saved research snapshot."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class ResearchFrontendAppTests(unittest.TestCase):
    def viewer(self):
        app = AppTest.from_file(str(ROOT / 'research_ui/viewer.py'), default_timeout=45).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def q_table(self, app):
        return next(frame.value for frame in app.dataframe if 'run_id' in frame.value.columns)

    def test_scope_and_downloads_are_visible_without_model_loading(self):
        app = self.viewer()
        self.assertEqual([item.value for item in app.metric], ['232', '226', '42', '428'])
        self.assertEqual(len(self.q_table(app)), 25)
        self.assertEqual(len(app.get('download_button')), 10)

    def test_dataset_widget_switches_to_goemotions_q_records(self):
        app = self.viewer()
        app.selectbox(key='research_dataset').select('go_emotions').run()
        self.assertEqual(len(app.exception), 0)
        table = self.q_table(app)
        self.assertEqual(len(table), 17)
        self.assertEqual(set(table.dataset), {'go_emotions'})
        self.assertEqual(set(table.metric), {'macro_ap'})

    def test_clearing_model_widget_shows_empty_selection(self):
        app = self.viewer()
        app.multiselect(key='q_models').set_value([]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('No fixed models match' in item.value for item in app.info))
        self.assertEqual(len(app.get('download_button')), 9)

    def test_stage_widget_selects_the_eight_s3_models(self):
        app = self.viewer()
        app.selectbox(key='q_stage').select('S3').run()
        self.assertEqual(len(app.exception), 0)
        table = self.q_table(app)
        self.assertEqual(len(table), 8)
        self.assertEqual(set(table.stage), {'S3'})

    def test_figure_and_case_widgets_render_after_selection(self):
        app = self.viewer()
        app.selectbox(key='research_figure').select(app.selectbox(key='research_figure').options[-1]).run()
        app.selectbox(key='research_case').select(app.selectbox(key='research_case').options[-1]).run()
        self.assertEqual(len(app.exception), 0)

    def test_legacy_prediction_entry_navigates_to_research(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        app.switch_page('pages/1_Current_Research.py').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, 'Limits and failures of text classification')


if __name__ == '__main__':
    unittest.main()
