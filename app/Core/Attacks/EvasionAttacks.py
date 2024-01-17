from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from Core import helper
from art.attacks.evasion import FastGradientMethod

class EvasionAttacks():
    def __init__(self,classifier, x_test: np.ndarray, y_test: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        inital_pred = self.classifier.predict(self.x_test)
        inital_acc = helper.calculate_accuracy(inital_pred,self.y_test)
        print("Accuracy on benign test examples: {}%".format(inital_acc * 100))

        attack_fgsm = FastGradientMethod(estimator=self.classifier, eps=0.2)
        x_test_adv_fgsm = attack_fgsm.generate(x=self.x_test)
        fgsm_pred = self.classifier.predict(x_test_adv_fgsm)
        fgsm_accuracy = helper.calculate_accuracy(fgsm_pred,self.y_test)
        print("Accuracy on FGSM test examples: {}%".format(fgsm_accuracy * 100))