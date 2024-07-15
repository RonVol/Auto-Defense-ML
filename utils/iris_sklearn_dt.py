from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import joblib

def get_model_path():
    # Define the model path relative to the current script
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    model_dir = os.path.join(current_dir, "../models")  # Path to the models directory
    model_dir = os.path.normpath(model_dir)  # Normalize the path
    model_path = os.path.join(model_dir, "iris_decision_tree.model")  # Full path to the model file
    return model_path

def get_x_test_path():
    # Define the model path relative to the current script
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    model_dir = os.path.join(current_dir, "../models")  # Path to the models directory
    model_dir = os.path.normpath(model_dir)  # Normalize the path
    model_path = os.path.join(model_dir, "iris_decision_tree_x_test.npy")  # Full path to the model file
    return model_path

def get_y_test_path():
    # Define the model path relative to the current script
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    model_dir = os.path.join(current_dir, "../models")  # Path to the models directory
    model_dir = os.path.normpath(model_dir)  # Normalize the path
    model_path = os.path.join(model_dir, "iris_decision_tree_y_test.npy")  # Full path to the model file
    return model_path

def get_y_test_proba_path():
    # Define the model path relative to the current script
    current_dir = os.path.dirname(__file__)  # Directory of the current script
    model_dir = os.path.join(current_dir, "../models")  # Path to the models directory
    model_dir = os.path.normpath(model_dir)  # Normalize the path
    model_path = os.path.join(model_dir, "iris_decision_tree_y_test_proba.npy")  # Full path to the model file
    return model_path

def create_iris_decision_tree():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Decision Tree classifier
    model = DecisionTreeClassifier()

    # Train the classifier
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model_path = get_model_path()
    joblib.dump(model, model_path)

    # Save x_test and y_test using NumPy
    np.save(get_x_test_path(), X_test)
    np.save(get_y_test_path(), y_test)
    np.save(get_y_test_proba_path(), y_pred_proba)

if __name__ == "__main__":
    create_iris_decision_tree()
