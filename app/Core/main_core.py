from art.estimators.classification import XGBoostClassifier
from xgboost import Booster, XGBClassifier
from app.data_loader import DataLoader
from app.config import supported_libraries, supported_attacks, supported_defenses
from art.attacks.evasion import ZooAttack
from app.Core.metrics_evaluator import MetricsEvaluator
from app.Core.attack_executor import AttackExecutor
from app.Core.defense_applier import DefenseApplier

class Main_Core:
    def __init__(self):
        self.dataloader = None
        self.classifier = None
        self.status = ""

    def get_status(self):
        """
        Returns the current status of the evaluation pipeline.
        
        :return: Current status as a string.
        """
        return self.status
    
    def setup_pipeline(self, dataloader: DataLoader, parameters, attacks, defenses):
        """
        Sets up the evaluation pipeline by initializing data, models, attacks, and defenses.
        
        :param dataloader: DataLoader instance for accessing the dataset.
        :param parameters: Dictionary containing model parameters and configuration.
        :param attacks: List of attack configurations to be evaluated.
        :param defenses: List of defense configurations to be applied.
        """
        self.dataloader = dataloader
        self.parameters = parameters
        self.attacks = attacks
        self.defenses = defenses
        self.setup_art_classifier()
        self.x_test = self.dataloader.x_test
        self.y_test = self.dataloader.y_test


    def main_loop(self, dataloader: DataLoader, parameters, attacks, defenses):
        """
        Main loop of the evaluation pipeline. Sets up the pipeline, applies defenses and attacks,
        and evaluates the model's performance.
        
        :param dataloader: DataLoader instance for accessing the dataset.
        :param parameters: Dictionary containing model parameters and configuration.
        :param attacks: List of attack configurations to be evaluated.
        :param defenses: List of defense configurations to be applied.
        :return: A dictionary containing all evaluation metrics.
        """
        self.status = "Setting up..."
        self.setup_pipeline(dataloader, parameters, attacks, defenses)

        all_metrics = {}
        # 1 - Evaluate on clean input
        clean_evaluator = MetricsEvaluator(self.classifier, self.x_test, self.y_test)
        all_metrics['clean'] = clean_evaluator

        for defense in defenses:
            applier = DefenseApplier(defense_config=defense,model=self.classifier)
            
            # 2 - Evaluate only on defense
            x_original_defended = applier.apply_defense(self.x_test)
            only_defense_evaluator = MetricsEvaluator(self.classifier, x_original_defended, self.y_test)
            all_metrics[defense['name']] = only_defense_evaluator

            for att in attacks:
                executor = AttackExecutor(attack_config=att, model=self.classifier)

                # 3 - Evaluate only on attack
                x_adv = executor.execute_attack(self.x_test)
                only_attack_evaluator = MetricsEvaluator(self.classifier, x_adv, self.y_test)
                all_metrics[att['name']] = only_attack_evaluator

                # 4 - Evaluate on defense and attack
                x_adv_defended = applier.apply_defense(x_adv)
                defense_attack_evaluator = MetricsEvaluator(self.classifier, x_adv_defended, self.y_test)
                all_metrics[defense['name'], att['name']] = defense_attack_evaluator

        self.print_all_metrics(all_metrics)
        return all_metrics
    
    def print_all_metrics(self, all_metrics, by_class=False):
        """
        Nicely prints out all the collected metrics from the evaluation pipeline.
        
        :param all_metrics: Dictionary containing all metrics, keyed by evaluation scenario.
        :param by_class: Indicates if metrics should be printed for each class individually.
        """
        for key, evaluator in all_metrics.items():
            # Determine the title based on the type of key (string or tuple)
            if isinstance(key, tuple):
                title = f"Metrics for Defense '{key[0]}' and Attack '{key[1]}'"
            else:
                title = f"Metrics for '{key}'"
            
            # Print the title
            print("="*len(title))
            print(title)
            print("="*len(title))
            
            # Use the print_metrics function of the MetricsEvaluator instance
            evaluator.print_metrics(by_class=by_class)
            print("\n")  # Add an empty line for better readability between section



    def setup_art_classifier(self):
        model = self.dataloader.model
        if isinstance(model,Booster) or isinstance(model, XGBoostClassifier):
            self.classifier = XGBoostClassifier(model=self.dataloader.model,
                                                    nb_features=self.parameters['nb_features'],
                                                    nb_classes=self.parameters['nb_classes'])
                


