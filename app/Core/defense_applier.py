from art.defences.preprocessor import SpatialSmoothing, JpegCompression, ThermometerEncoding, FeatureSqueezing
# Import other attacks as needed

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
        if defense_name == 'FeatureSqueezing':
            bit_depth = self.defense_config.get('bit_depth')
            apply_fit = self.defense_config.get('apply_fit')
            apply_predict = self.defense_config.get('apply_predict')

            defense =  FeatureSqueezing(bit_depth=bit_depth,apply_fit=apply_fit,apply_predict=apply_predict,clip_values=self.clip_values)
            return defense
        
        elif defense_name == 'JpegCompression':
            clip_values = self.defense_config.get('clip_values')
            quality = self.defense_config.get('quality')
            apply_fit = self.defense_config.get('apply_fit')
            apply_predict = self.defense_config.get('apply_predict')

            defense =  JpegCompression(clip_values=clip_values,quality=quality,apply_fit=apply_fit,apply_predict=apply_predict)
            return defense
        
        elif defense_name == 'ThermometerEncoding':
            clip_values = self.defense_config.get('clip_values')
            apply_fit = self.defense_config.get('apply_fit')
            apply_predict = self.defense_config.get('apply_predict')

            defense =  ThermometerEncoding(clip_values=clip_values,apply_fit=apply_fit,apply_predict=apply_predict)
            return defense
        else:
            raise ValueError(f"Unsupported defense: {defense_name}")

    def apply_defense(self, x):
        """
        Applies the initialized defense to the input data.
        
        :param x: The input data to which the defense will be applied. Expected to be in a flattened format.
        :return: The defended input data, reshaped back to its original flattened format.
        """
        #print(f"x before:{x}")
        x_defended,_ = self.defense(x)
        #print(f"x after:{x_defended}")
        return x_defended
