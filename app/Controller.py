from .UI.main_ui import UI
from .Core.main_core import Main_Core
import xgboost as xgb

class Controller:
    def __init__(self, core : Main_Core, ui : UI):
        self.core = core
        self.ui = ui

    def run_ui(self):
        self.ui.run_ui()

    def get_user_input(self):
        self.inp = self.ui.get_simulated_user_input()
        if not self.load_model:
            # send to ui that failed to load model
            print("...")

    def load_model(self):
        model_fpath = self.inp['model_path']
        # if .npy,if .mode..if pickle...
        try:
            model = xgb.Booster()
            self.model = model.load_model(model_fpath)  
            print(f"Model loaded from {model_fpath}")
            return True
        except:
            print(f"Failed to load model from {model_fpath}")
        return False

