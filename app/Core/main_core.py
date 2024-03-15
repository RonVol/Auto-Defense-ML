from art.estimators.classification import XGBoostClassifier
from xgboost import Booster, XGBClassifier
from app.data_loader import DataLoader
from app.config import supported_libraries, supported_attacks, supported_defenses
from art.attacks.evasion import ZooAttack
from app.Core.metrics_evaluator import MetricsEvaluator
from app.Core.attack_executor import AttackExecutor
class Main_Core:

    def __init__(self):
        self.dataloader = None
        self.classifier = None
        self.status = ""

    def get_status(self):
        return self.status

    def run_pipeline(self, dataloader: DataLoader, parameters, attacks, defenses):
        self.status = "Starting run..."
        self.dataloader = dataloader
        self.parameters = parameters
        self.attacks = attacks
        self.defenses = defenses

        self.status = "Setting up classifier..."
        self.SetupArtClassifier()
        
        self.status = "Predicting on clean test set..."

        x_test = self.dataloader.x_test
        y_test = self.dataloader.y_test

        # clean_run_evaluator = MetricsEvaluator(self.classifier, self.dataloader.x_test, self.dataloader.y_test)
        # clean_run_metrics = clean_run_evaluator.run_metrics_calculations()
        # clean_run_evaluator.print_metrics(by_class=False)

        all_attack_metrics = {}
        for att in attacks:
            executor = AttackExecutor(attack_config=att, model=self.classifier, x_test=x_test, y_test=y_test)
            x_adv = executor.execute_attack()
            att_evaluator = MetricsEvaluator(self.classifier, x_adv, y_test)
            att_metrics = att_evaluator.run_metrics_calculations()
            all_attack_metrics[att['name']] = att_metrics
        print(all_attack_metrics)

    def SetupArtClassifier(self):
        model = self.dataloader.model
        if isinstance(model,Booster) or isinstance(model, XGBoostClassifier):
            self.classifier = XGBoostClassifier(model=self.dataloader.model,
                                                    nb_features=self.parameters['nb_features'],
                                                    nb_classes=self.parameters['nb_classes'])
                


