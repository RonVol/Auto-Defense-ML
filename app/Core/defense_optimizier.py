from skopt import gp_minimize
from skopt.space import Real, Integer
from app.Core.metrics_evaluator import MetricsEvaluator
from app.Core.defense_applier import DefenseApplier
from app.data_loader import DataLoader
import numpy as np

class DefensekOptimizier:
    def __init__(self,defense, dataloader, classifier, logger=None):
        self.defense = defense
        self.__dataloader = dataloader
        self.__classifier = classifier
        self.set_space()
        self.alpha = 0.5  # Weight for the accuracy term
        self.beta = 0.5   # Weight for the perturbation term
        self.logger = logger
        self.logger.info(f"OPTIMIZIER::INIT")

    def set_space(self):
        # Define a mapping of attack names to parameter names
        self.defense_param_mapping = {
            'FeatureSqueezing': ["bit_depth"]
        }
        defense_name = self.defense['name']
        if defense_name == 'FeatureSqueezing':
            self.space = [
                Integer(3, 9, name="bit_depth")  
            ]
        
        else:
            raise ValueError(f"Unsupported defense for optimization: {defense_name}")

    def validate_parameters_format(self, params):
        if "bit_depth" in params:
                params["bit_depth"] = int(params["bit_depth"])

        return params

    def update_logger(self,config,params, perturbation,accuracy):
         defense_name = self.defense['name']
         if defense_name in self.defense_param_mapping:
              param_names = self.defense_param_mapping[defense_name]
              defense_params = dict(zip(param_names, params))
         self.logger.info(f"OPTIMIZIER::{defense_name}::{defense_params}::perturbation={perturbation}, accuracy={accuracy}, objective={self.alpha*perturbation+self.beta*accuracy}")

         

    def optimize(self):
        # Define the objective function to be minimized
        defense_name = self.defense['name']
        def objective(params):
            defense_config = self.defense.copy()      
            if defense_name in self.defense_param_mapping:
                param_names = self.defense_param_mapping[defense_name]
                defense_params = dict(zip(param_names, params))

                defense_params = self.validate_parameters_format(defense_params)

                defense_config.update(defense_params)
            else:
                raise ValueError(f"Unsupported defense for optimization: {defense_name}")
            
            try:
                applier = DefenseApplier(defense_config, self.__classifier, self.__dataloader.clip_values)
                x_defended = applier.apply_defense(x=self.__dataloader.x)
                evaluator = MetricsEvaluator(self.__classifier, x_defended, self.__dataloader.y)
                metrics = evaluator.get_metrics()

                # Calculate the perturbation size 
                perturbation = np.mean(np.linalg.norm(self.__dataloader.x - x_defended, axis=1))         
                # Combine the objectives
                combined_objective = 1 - metrics['overall_accuracy']
                self.update_logger(defense_config,params,perturbation,metrics['overall_accuracy'] )
                return combined_objective
            except Exception as e:
                # Handle potential errors during attack execution and evaluation
                print(f"Error during defense optimization execution: {e}")
                return float('inf')  # Return a large value to indicate failure
            
            

        # Perform Bayesian optimization
        result = gp_minimize(objective, self.space, n_calls=10, random_state=0)
        
        optimized_params = dict(zip(self.defense_param_mapping[defense_name], result.x))
        optimized_params = self.validate_parameters_format(optimized_params)
        self.logger.info(f"OPTIMIZIER-FINAL::{defense_name}::{optimized_params}::Objective={result.fun}::models={result.models}")

        optimized_defense = self.defense.copy()
        optimized_defense.update(optimized_params)
        return optimized_defense