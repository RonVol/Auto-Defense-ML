from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
import numpy as np

class MetricsEvaluator:
    def __init__(self, model, x_test, y_test):
        self.classifier = model
        self.x_test = x_test
        # Convert y_test from one-hot encoding to label encoding
        self.y_test = np.argmax(y_test, axis=1)
        self.y_pred = self.predict()

    def predict(self):
        # Assuming the classifier has a predict method that returns class labels
        # If your classifier returns probabilities (for each class), you'll also need to convert these to class labels
        y_pred = self.classifier.predict(self.x_test)
        if y_pred.shape[1] > 1:  # Check if y_pred is probabilities (one-hot encoded)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    
    def run_metrics_calculations(self):
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
        return classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)
    
    def print_metrics(self,by_class):
        print("Overall Accuracy:", self.overall_accuracy)
        print("Overall Recall:", self.overall_precision)
        print("Overall Precision:", self.overall_recall)
        if by_class:
            print("Metrics by Class:\n", self.metrics_per_class)
