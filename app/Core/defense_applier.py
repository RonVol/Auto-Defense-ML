from art.defences.preprocessor import FeatureSqueezing
from art.defences.postprocessor.class_labels import ClassLabels
from app.Core.attacks.MonteCarloClassifier import MonteCarloDecisionTreeClassifier, MonteCarloRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import copy

class DefenseApplier:
    """
    The DefenseApplier class is responsible for applying defenses to input data.
    """
    def __init__(self, defense_config, model, clip_values):
        """
        Initializes the DefenseApplier with a defense configuration and a model.
        
        :param defense_config: A dictionary containing the configuration for the defense to be applied. 
                               This includes the name of the defense and any necessary parameters.
        :param model: The machine learning model to which the defense will be applied. 
                      This parameter is currently not directly used but can be useful for future extensions where
                      defense application might depend on model-specific characteristics.
        """
        self.defense_config = defense_config
        self.model = model
        self.clip_values = clip_values
        self.defense = self.initialize_defense()

    def initialize_defense(self):
        """
        Initializes the defense based on the configuration provided during object creation.
        
        :return: An instance of the specified defense.
        """
        defense_name = self.defense_config['name']
        print(f"\n\nIN INIT DEFENSE:{self.model}\n\n")
        if self.defense_config['defense_type'] == "new_classifier":
            if defense_name == 'TTTS':
                prob_type = self.defense_config.get('prob_type')
                n_simulations = self.defense_config.get('n_simulations')
                new_classifier = self.get_ttts_class(prob_type=prob_type, n_simulations=n_simulations)
                new_classifier.__dict__.update(copy.deepcopy(self.model.__dict__))
                return new_classifier

        # pre/post processor defenses
        if defense_name == 'FeatureSqueezing':
            bit_depth = self.defense_config.get('bit_depth')
            apply_fit = self.defense_config.get('apply_fit')
            apply_predict = self.defense_config.get('apply_predict')

            defense =  FeatureSqueezing(bit_depth=bit_depth,apply_fit=apply_fit,apply_predict=apply_predict,clip_values=self.clip_values)
            return defense
        elif defense_name == 'ClassLabels':
            apply_fit = self.defense_config.get('apply_fit')
            apply_predict = self.defense_config.get('apply_predict')
            return ClassLabels(apply_fit=apply_fit,apply_predict=apply_predict)
        else:
            raise ValueError(f"Unsupported defense: {defense_name}")
    
    def get_ttts_class(self, prob_type, n_simulations=None):
        if isinstance(self.model, DecisionTreeClassifier):
            return MonteCarloDecisionTreeClassifier(prob_type=prob_type, n_simulations=n_simulations)
        elif isinstance(self.model, RandomForestClassifier):
            return MonteCarloRandomForestClassifier(prob_type=prob_type)
        
    def apply_preprocessor(self, x):
        x_defended, _ = self.defense(x)
        return x_defended
    
    def apply_postprocessor(self, y_pred):
        return self.defense(y_pred)
    
    def is_preprocessor(self):
        print(f"in is_preprocessor :{self.defense_config['defense_type']}")
        try:
            if self.defense_config['defense_type'] == "preprocessor" : 
                return True         
            else:
                return False
        except Exception as e:
                print(f"Error in defense_applier.is_preprocessor(): {e}")

    def apply_defense(self, x=None, y_pred=None):
        if self.is_preprocessor():
            return self.apply_preprocessor(x)
        else:
            return self.apply_postprocessor(y_pred) # postprocessor
