# import unittest
# from unittest.mock import MagicMock
# from app.Core.main_core import Main_Core
# from app.data_loader import DataLoader
# from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier

# class TestMainCore(unittest.TestCase):

#     def setUp(self):
#         self.main_core = Main_Core()

#     def test_initial_status(self):
#         self.assertEqual(self.main_core.status, "Idle")

#     def test_dataloader_setter(self):
#         mock_dataloader = MagicMock(spec=DataLoader)
#         mock_dataloader.model = MagicMock(spec=XGBClassifier)
#         mock_dataloader.nb_features = 10
#         mock_dataloader.nb_classes = 2
        
#         self.main_core.dataloader = mock_dataloader
#         self.assertEqual(self.main_core.dataloader, mock_dataloader)

#     def test_dataloader_setter_invalid(self):
#         with self.assertRaises(ValueError):
#             self.main_core.dataloader = "Invalid DataLoader"

#     def test_perform_attacks_no_dataloader(self):
#         with self.assertRaises(ValueError):
#             self.main_core.perform_attacks([])

#     def test_perform_defenses_no_dataloader(self):
#         with self.assertRaises(ValueError):
#             self.main_core.perform_defenses([])

#     def test_perform_defenses_on_attacks_no_dataloader(self):
#         with self.assertRaises(ValueError):
#             self.main_core.perform_defenses_on_attacks([], {})

#     def test_perform_benign_evaluation_no_dataloader(self):
#         with self.assertRaises(ValueError):
#             self.main_core.perform_benign_evaluation()

# if __name__ == '__main__':
#     unittest.main()
