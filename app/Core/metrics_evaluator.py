from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
import numpy as np

class MetricsEvaluator:
    """
    The MetricsEvaluator class is designed to compute and store various performance metrics for a given machine learning model.
    """
    def __init__(self, model, x_test, y_test):
        """
        Initializes the MetricsEvaluator with a model and test dataset.
        
        :param model: The machine learning model to be evaluated.
        :param x_test: Test dataset features.
        :param y_test: Test dataset labels, assumed to be one-hot encoded.
        """
        self.classifier = model
        self.x_test = x_test
        # Convert y_test from one-hot encoding to label encoding
        self.y_test = np.argmax(y_test, axis=1)
        self.y_pred = self.predict()
        self.run_metrics_calculations()

    def predict(self):
        """
        Predicts the labels for the test dataset using the provided model.
        
        :return: The predicted labels, converted from one-hot encoding if necessary.
        """
        y_pred = self.classifier.predict(self.x_test)
        if y_pred.shape[1] > 1:  # Check if y_pred is probabilities (one-hot encoded)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    
    def run_metrics_calculations(self):
        """
        Calculates various performance metrics based on the true labels and the predictions.
        
        :return: A dictionary containing overall accuracy, precision, recall, and metrics per class.
        """
        self.overall_accuracy = accuracy_score(self.y_test, self.y_pred)
        self.overall_precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        self.overall_recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        self.metrics_per_class = self.calculate_metrics_per_class() # might need to extend this for other metrics, so different function
        return {
            "overall_accuracy":self.overall_accuracy,
            "overall_precision":self.overall_precision,
            "overall_recall":self.overall_recall,
            "metrics_per_class":self.metrics_per_class,
        }


    def calculate_metrics_per_class(self):
        """
        Calculates precision, recall, and f1-score for each class individually.
        
        :return: A dictionary with the classification report for each class.
        """
        return classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)
    
    def print_metrics(self,by_class):
        """
        Prints the calculated metrics. Includes overall accuracy, precision, recall, and optionally metrics per class.
        
        :param by_class: If True, prints detailed metrics for each class.
        """
        print("Overall Accuracy:", self.overall_accuracy)
        print("Overall Recall:", self.overall_precision)
        print("Overall Precision:", self.overall_recall)
        if by_class:
            print("Metrics by Class:\n", self.metrics_per_class)
