import unittest
import numpy as np
from unittest.mock import MagicMock
from app.Core.main_core import Main_Core
from app.data_loader import DataLoader
import app.config as config
import os
import xgboost as xgb

class TestMainCore(unittest.TestCase):

    def setUp(self):
        self.main_core = Main_Core()

        # Load actual data
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        x_test = np.load(os.path.join(models_path, 'iris_xgboost_x_test.npy'))
        y_test = np.load(os.path.join(models_path, 'iris_xgboost_y_test.npy'))
        model = xgb.Booster()
        model.load_model(os.path.join(models_path, 'iris_xgboost.model'))

        # Limit the dataset size for faster execution
        sample_size = 10 
        x_test = x_test[:sample_size]
        y_test = y_test[:sample_size]

        # Create a mock DataLoader with the actual data
        self.mock_dataloader = MagicMock(spec=DataLoader)
        self.mock_dataloader.model = model
        self.mock_dataloader.nb_features = x_test.shape[1]
        self.mock_dataloader.nb_classes = len(np.unique(y_test))
        self.mock_dataloader.x = x_test
        self.mock_dataloader.y = y_test
        self.mock_dataloader.clip_values = (0.0, 1.0)

        self.main_core.dataloader = self.mock_dataloader

    # TC_Core_01
    def test_initial_status(self):
        self.assertEqual(self.main_core.status, "Idle")

    # TC_Core_01
    def test_dataloader_setter(self):
        self.assertEqual(self.main_core.dataloader, self.mock_dataloader)

    # TC_Core_01
    def test_dataloader_setter_invalid(self):
        with self.assertRaises(ValueError):
            self.main_core.dataloader = "Invalid DataLoader"

    # TC_Core_02
    def test_optimize_attacks(self):
        attacks = [config.supported_attacks["ZooAttack"], config.supported_attacks["HopSkipJump"]]
        optimized_attacks = self.main_core.optimize_attacks(attacks)
        self.assertIsInstance(optimized_attacks, list)
        self.assertEqual(len(optimized_attacks), len(attacks))

    # TC_Core_03
    def test_perform_attacks(self):
        attacks = [config.supported_attacks["ZooAttack"], config.supported_attacks["HopSkipJump"]]
        metrics, adv_examples = self.main_core.perform_attacks(attacks)
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(adv_examples, dict)

    # TC_Core_04
    def test_perform_defenses(self):
        defenses = [config.supported_defenses["FeatureSqueezing"], config.supported_defenses["ClassLabels"]]
        metrics, defended_examples = self.main_core.perform_defenses(defenses)
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(defended_examples, dict)

    # TC_Core_05
    def test_perform_defenses_on_attacks(self):
        defenses = [config.supported_defenses["FeatureSqueezing"], config.supported_defenses["ClassLabels"]]
        adv_examples = {'ZooAttack': np.random.rand(*self.mock_dataloader.x.shape), 'HopSkipJump': np.random.rand(*self.mock_dataloader.x.shape)}
        metrics, adv_defended_examples = self.main_core.perform_defenses_on_attacks(defenses, adv_examples)
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(adv_defended_examples, dict)

    # TC_Core_06
    def test_perform_benign_evaluation(self):
        metrics = self.main_core.perform_benign_evaluation()
        self.assertIsInstance(metrics, dict)

    # TC_Core_07
    def test_optimize_defenses(self):
        defenses = [config.supported_defenses["FeatureSqueezing"]]
        optimized_defenses = self.main_core.optimize_defenses(defenses)
        self.assertIsInstance(optimized_defenses, list)
        self.assertEqual(len(optimized_defenses), len(defenses))

    # TC_Core_08
    def test_optimize_attacks_invalid(self):
        invalid_attack = {'name': 'InvalidAttack'}
        with self.assertRaises(ValueError):
            self.main_core.optimize_attacks([invalid_attack])

    # TC_Core_09
    def test_optimize_defenses_invalid(self):
        invalid_defense = {'name': 'InvalidDefense'}
        with self.assertRaises(ValueError):
            self.main_core.optimize_defenses([invalid_defense])

    # TC_Core_10
    def test_perform_attacks_no_attacks(self):
        metrics, adv_examples = self.main_core.perform_attacks([])
        self.assertEqual(metrics, {})
        self.assertEqual(adv_examples, {})

    # TC_Core_11
    def test_perform_defenses_no_defenses(self):
        metrics, defended_examples = self.main_core.perform_defenses([])
        self.assertEqual(metrics, {})
        self.assertEqual(defended_examples, {})

    # TC_Core_12
    def test_perform_benign_evaluation_no_data(self):
        self.mock_dataloader.x = np.array([])
        self.mock_dataloader.y = np.array([])
        with self.assertRaises(ValueError):
            metrics = self.main_core.perform_benign_evaluation()

if __name__ == '__main__':
    unittest.main()
