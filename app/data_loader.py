import xgboost as xgb
import numpy as np

class DataLoader:
    def __init__(self):
        self.model = None
        self.x_test = None
        self.y_test = None

    def load_model(self,library, path_to_model):
        #switch case on whitch library...
        #if library is xgboost...
        try:
            self.model = xgb.Booster()
            self.model.load_model(path_to_model)
            print(f"Model loaded from {path_to_model}")
            return True
        except:
            print(f"Failed to load model from {path_to_model}")
        return False
    
    def load_test(self,x_test_path=None,y_test_path=None):

        # if .npy extension...
        if x_test_path is not None:
            self.x_test = np.load(x_test_path)
        if y_test_path is not None:
            self.y_test = np.load(y_test_path)

