import xgboost as xgb
import numpy as np
import joblib

class DataLoader:
    def __init__(self):
        self.__model = None
        self.__x_test = None
        self.__y_test = None
        self.__y_test_proba = None

    @property
    def model(self):
        return self.__model
    
    
    @property
    def x(self):
        return self.__x_test
    
    @property
    def y(self):
        return self.__y_test

    def load_model(self,library, path_to_model) -> bool:
        lib_name = library['name']
        try:
            if lib_name == 'XGBoost':
                self.__model = xgb.XGBClassifier()
                self.__model.load_model(path_to_model)
                print(f"XGBoost model loaded from {path_to_model}")
            elif lib_name == 'scikit-learn':
                self.__model = joblib.load(path_to_model)
                print(f"scikit-learn model loaded from {path_to_model}")
            else:
                print(f"Unsupported library: {lib_name}")
                return False
            return True
        except Exception as e:
            print(f"Failed to load model from {path_to_model}. Error: {str(e)}")
            return False
    
    def load_test(self,x_test_path=None,y_test_path=None, y_test_proba_fpath=None):
        # if .npy extension...
        try:
            if x_test_path is not None:
                self.__x_test = np.load(x_test_path)
            if y_test_path is not None:
                self.__y_test = np.load(y_test_path)
            if y_test_proba_fpath is not None:
                self.__y_test_proba = np.load(y_test_proba_fpath)
            return True
        except:
            return False


    @property
    def nb_classes(self):
        return len(np.unique(self.__y_test))
    
    @property
    def nb_features(self):
        return self.__x_test.shape[1]
    
    @property
    def clip_values(self):
        min_clip = self.__x_test.min(axis=0).min()  # Minimum values for each feature across all samples
        max_clip = self.__x_test.max(axis=0).max()  # Maximum values for each feature across all samples
        clip_values = (min_clip, max_clip)
        return clip_values



