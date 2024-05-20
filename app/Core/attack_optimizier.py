from skopt import gp_minimize
from skopt.space import Real, Integer
from app.Core.metrics_evaluator import MetricsEvaluator
from app.Core.attack_executor import AttackExecutor
from app.data_loader import DataLoader
import numpy as np

class AttackOptimizier:
    def __init__(self,attack, dataloader, classifier, logger=None):
        self.attack = attack
        self.__dataloader = dataloader
        self.__classifier = classifier
        self.set_space()
        self.alpha = 0.5  # Weight for the accuracy term
        self.beta = 0.5   # Weight for the perturbation term
        self.logger = logger
        self.logger.info(f"OPTIMIZIER::INIT")

    def set_space(self):
        # Define a mapping of attack names to parameter names
        self.attack_param_mapping = {
            'ZooAttack': ["learning_rate", "initial_const", "max_iter"],
            'HopSkipJump': ["init_size", "init_eval", "max_iter"],
            'SignOPTAttack': ["epsilon", "max_iter"],
            'BoundaryAttack': ["epsilon", "max_iter"]
        }
        attack_name = self.attack['name']
        if attack_name == 'ZooAttack':
            self.space = [
                Real(1e-3, 1e-1, name="learning_rate"),
                Real(1e-5, 1e-2, name="initial_const"),
                Integer(10, 50, name="max_iter")  
            ]
        
        elif attack_name == 'HopSkipJump':
            self.space = [
                Integer(50, 100, name="init_size"),
                Integer(50, 100, name="init_eval"),# needs to be < than max_eval
                Integer(50, 150, name="max_iter")  
            ]
        
        elif attack_name == 'SignOPTAttack':
            self.space = [
                Real(1e-4, 1e-2, name="epsilon"),
                Integer(5, 15, name="max_iter") 
            ]
        
        elif attack_name == 'BoundaryAttack':
            self.space = [
                Real(1e-3, 1e-1, name="epsilon"),
                Integer(80, 120, name="max_iter")
            ]
        else:
            raise ValueError(f"Unsupported attack for optimization: {attack_name}")

    def validate_parameters_format(self, params):
        if "max_iter" in params:
                params["max_iter"] = int(params["max_iter"])
        if "init_size" in params:
                params["init_size"] = int(params["init_size"])
        if "init_eval" in params:
                params["init_eval"] = int(params["init_eval"])

        return params

    def update_logger(self,config,params, perturbation,accuracy):
         attack_name = self.attack['name']
         if attack_name in self.attack_param_mapping:
              param_names = self.attack_param_mapping[attack_name]
              attack_params = dict(zip(param_names, params))
         self.logger.info(f"OPTIMIZIER::{attack_name}::{attack_params}::perturbation={perturbation}, accuracy={accuracy}, objective={self.alpha*perturbation+self.beta*accuracy}")

         

    def optimize(self):
        # Define the objective function to be minimized
        attack_name = self.attack['name']
        def objective(params):
            attack_config = self.attack.copy()      
            if attack_name in self.attack_param_mapping:
                param_names = self.attack_param_mapping[attack_name]
                attack_params = dict(zip(param_names, params))

                attack_params = self.validate_parameters_format(attack_params)

                attack_config.update(attack_params)
            else:
                raise ValueError(f"Unsupported attack for optimization: {attack_name}")
            
            try:
                executor = AttackExecutor(attack_config, self.__classifier, self.__dataloader.clip_values)
                x_adv = executor.execute_attack(self.__dataloader.x)
                evaluator = MetricsEvaluator(self.__classifier, x_adv, self.__dataloader.y)
                metrics = evaluator.get_metrics()

                # Calculate the perturbation size 
                perturbation = np.mean(np.linalg.norm(self.__dataloader.x - x_adv, axis=1))         
                # Combine the objectives
                combined_objective = self.alpha * metrics['overall_accuracy'] + self.beta * perturbation
                self.update_logger(attack_config,params,perturbation,metrics['overall_accuracy'] )
                return combined_objective
            except Exception as e:
                # Handle potential errors during attack execution and evaluation
                print(f"Error during attack optimization execution: {e}")
                return float('inf')  # Return a large value to indicate failure
            
            

        # Perform Bayesian optimization
        result = gp_minimize(objective, self.space, n_calls=10, random_state=0)
        
        optimized_params = dict(zip(self.attack_param_mapping[attack_name], result.x))
        optimized_params = self.validate_parameters_format(optimized_params)
        self.logger.info(f"OPTIMIZIER-FINAL::{attack_name}::{optimized_params}::Objective={result.fun}::models={result.models}")

        optimized_attack = self.attack.copy()
        optimized_attack.update(optimized_params)
        return optimized_attack
