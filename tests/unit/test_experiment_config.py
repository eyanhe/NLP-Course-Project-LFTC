"""Experiment grids preserve configuration values and deterministic order."""
import unittest


class ExperimentConfigTests(unittest.TestCase):
    def test_grid_expands_datasets_models_seeds_and_conditions(self):
        from lftc.config import expand_experiment_config
        configs = expand_experiment_config({
            "schema_version": 1, "name": "comparison", "datasets": ["imdb", "mnli"],
            "models": ["tfidf_lr", "bilstm"], "seeds": [42, 43],
            "conditions": [{"mode": "all"}, {"mode": "kshot", "k": 4}],
            "training": {"epochs": 2},
        })
        self.assertEqual(len(configs), 16)
        self.assertEqual(configs[0]["dataset"], "imdb")
        self.assertEqual(configs[0]["model"], "tfidf_lr")
        self.assertEqual(configs[0]["sampling"], {"mode": "all"})
        self.assertEqual(configs[-1]["sampling"], {"mode": "kshot", "k": 4})
        configs[0]["training"]["epochs"] = 9
        self.assertEqual(configs[1]["training"]["epochs"], 2)

    def test_empty_grid_and_unknown_model_are_configuration_errors(self):
        from lftc.config import expand_experiment_config
        from lftc.domain import ConfigError
        for config in ({"datasets": [], "models": ["tfidf_lr"]},
                       {"dataset": "imdb", "model": "unknown"}):
            with self.subTest(config=config), self.assertRaises(ConfigError):
                expand_experiment_config(config)
