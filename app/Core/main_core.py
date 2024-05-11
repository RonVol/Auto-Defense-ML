from art.estimators.classification import XGBoostClassifier
from xgboost import Booster, XGBClassifier
import xgboost as xgb
from app.data_loader import DataLoader
from app.config import supported_libraries, supported_attacks, supported_defenses
from art.attacks.evasion import ZooAttack
from app.Core.metrics_evaluator import MetricsEvaluator
from app.Core.attack_executor import AttackExecutor
from app.Core.defense_applier import DefenseApplier
from app.Core.attack_optimizier import AttackOptimizier
import logging


class Main_Core:
    def __init__(self):
        self.logger = self.setup_logger()
        self.__dataloader = None
        self.__status = "Idle"
        self.__classifier = None
    
    def setup_logger(self):
        logger = logging.getLogger("Main_Core")
        logger.setLevel(logging.INFO)

        # Create a file handler and set the log file
        file_handler = logging.FileHandler("Main_Core_log.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    @property
    def dataloader(self):
        """The getter method for the dataloader property."""
        return self.__dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        """The setter method for the dataloader property."""
        if not isinstance(dataloader, DataLoader):
            raise ValueError("dataloader must be an instance of DataLoader")
        self.__dataloader = dataloader
        self.setup_art_classifier() # sets self.classifier wrapped in ART classifier

    @property
    def status(self):
        return self.__status
    
    def requires_dataloader(func):
        def wrapper(self, *args, **kwargs):
            if self.__dataloader is None or self.__classifier is None:
                raise ValueError("Dataloader has not been set.")
            return func(self, *args, **kwargs)
        return wrapper
    
    def optimize_attacks(self, attacks):
        print("in optimize attacks")

        optimized_attacks = []
        for attack in attacks:
            att_optimizer = AttackOptimizier(attack,self.__dataloader,self.__classifier)
            optimized_att = att_optimizer.optimize()
            optimized_attacks.append(optimized_att)
            self.logger.info(f"Optimized parameters: {optimized_att}")
        return optimized_attacks
    
    @requires_dataloader
    def perform_attacks(self, attacks):
        x_org = self.__dataloader.x
        y_org = self.__dataloader.y
        clip_values = self.__dataloader.clip_values
        metrics = {}
        adv_examples = {}
        for att in attacks:
            executor = AttackExecutor(attack_config=att, model=self.__classifier,clip_values=clip_values)
            x_adv = executor.execute_attack(x_org)
            evaluator = MetricsEvaluator(self.__classifier, x_adv, y_org)
            metrics[att['name']] = evaluator.get_metrics()
            adv_examples[att['name']] = x_adv
        return metrics, adv_examples
    
    @requires_dataloader
    def perform_defenses(self, defenses):
        x_org = self.__dataloader.x
        y_org = self.__dataloader.y
        clip_values = self.__dataloader.clip_values
        metrics = {}
        defended_examples = {}
        for defense in defenses:

            applier = DefenseApplier(defense_config=defense, model=self.__classifier,clip_values=clip_values)
            x_defended = applier.apply_defense(x_org)
            evaluator = MetricsEvaluator(self.__classifier, x_defended, y_org)
            metrics[defense['name']] = evaluator.get_metrics()
            defended_examples[defense['name']] = x_defended
        return metrics, defended_examples
    
    @requires_dataloader
    def perform_defenses_on_attacks(self, defenses, adv_examples):
        y_org = self.__dataloader.y
        clip_values = self.__dataloader.clip_values
        metrics = {}
        adv_defended_examples = {}
        for defense in defenses:
            applier = DefenseApplier(defense_config=defense, model=self.__classifier,clip_values=clip_values)
            for att_name, adv_ex in adv_examples.items():         
                x_adv_defended = applier.apply_defense(adv_ex)
                evaluator = MetricsEvaluator(self.__classifier, x_adv_defended, y_org)
                metrics[defense['name'], att_name] = evaluator.get_metrics()
                adv_defended_examples[defense['name'], att_name] = x_adv_defended

        return metrics, adv_defended_examples
    
    @requires_dataloader
    def perform_benign_evaluation(self):
        x_org = self.__dataloader.x
        y_org = self.__dataloader.y
        metrics = {}
        evaluator = MetricsEvaluator(self.__classifier, x_org, y_org)
        metrics['Clean'] = evaluator.get_metrics()
        return metrics
    
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
        model = self.__dataloader.model
        nb_features=self.__dataloader.nb_features
        nb_classes=self.__dataloader.nb_classes
        if isinstance(model,Booster) or isinstance(model, XGBoostClassifier) or isinstance(model, xgb.XGBClassifier):
            self.__classifier = XGBoostClassifier(model=model, nb_features=nb_features, nb_classes=nb_classes)



                


