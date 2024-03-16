from art.defences.preprocessor import SpatialSmoothing, JpegCompression, ThermometerEncoding
# Import other attacks as needed

class DefenseApplier:
    def __init__(self, defense_config, model):
        self.defense_config = defense_config
        self.model = model
        self.defense = self.initialize_defense()

    def initialize_defense(self):
        defense_name = self.defense_config['name']
        if defense_name == 'SpatialSmoothing':
            window_size = self.defense_config.get('window_size')
            apply_fit = self.defense_config.get('apply_fit')
            apply_predict = self.defense_config.get('apply_predict')

            defense =  SpatialSmoothing(window_size=window_size,apply_fit=apply_fit,apply_predict=apply_predict)
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
        x_reshaped = x.reshape((-1, 28, 28, 1))
        x_defended,_ = self.defense(x_reshaped)
        x_defended = x_defended.reshape(-1, 28*28)
        return x_defended
