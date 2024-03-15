import xgboost as xgb
import numpy as np
from art.utils import load_mnist
import os


def get_model_path():
    # Define the model path relative to the current script
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    model_dir = os.path.join(current_dir, "../models")  # Path to the models directory
    model_dir = os.path.normpath(model_dir)  # Normalize the path
    model_path = os.path.join(model_dir, "mnist_xgboost.model")  # Full path to the model file
    return model_path

def get_x_test_path():
    # Define the model path relative to the current script
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    model_dir = os.path.join(current_dir, "../models")  # Path to the models directory
    model_dir = os.path.normpath(model_dir)  # Normalize the path
    model_path = os.path.join(model_dir, "mnist_xgboost_x_test.npy")  # Full path to the model file
    return model_path

def get_y_test_path():
    # Define the model path relative to the current script
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    model_dir = os.path.join(current_dir, "../models")  # Path to the models directory
    model_dir = os.path.normpath(model_dir)  # Normalize the path
    model_path = os.path.join(model_dir, "mnist_xgboost_y_test.npy")  # Full path to the model file
    return model_path

    

def create_mnist_XGboost():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    # Flatten dataset
    x_test = x_test[0:5]
    y_test = y_test[0:5]

    nb_samples_train = x_train.shape[0]
    nb_samples_test = x_test.shape[0]
    x_train = x_train.reshape((nb_samples_train, 28 * 28))
    x_test = x_test.reshape((nb_samples_test, 28 * 28))

    # Create the model
    params = {"objective": "multi:softprob", "eval_metric": ["mlogloss", "merror"], "num_class": 10}
    dtrain = xgb.DMatrix(x_train, label=np.argmax(y_train, axis=1))
    dtest = xgb.DMatrix(x_test, label=np.argmax(y_test, axis=1))
    evals = [(dtest, "test"), (dtrain, "train")]
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=2, evals=evals)

    #save the model 
    model_path = get_model_path()
    model.save_model(model_path)
    # Save x_test and y_test using NumPy
    np.save(get_x_test_path(), x_test)
    np.save(get_y_test_path(), y_test)
    return model

def save_only_tests():
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    nb_samples_test = x_test.shape[0]
    x_test = x_test.reshape((nb_samples_test, 28 * 28))
    np.save(get_x_test_path(), x_test[:10])
    np.save(get_y_test_path(), y_test[:10])


def load_mnist_XGBoost():
    # Load the model
    model_path = get_model_path()
    model = xgb.Booster()
    model.load_model(model_path)
    
    print(f"Model loaded from {model_path}")
    return model

if __name__ == "__main__":
    #create_mnist_XGboost()
    save_only_tests()