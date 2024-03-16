from art.attacks.evasion import ZooAttack, HopSkipJump, SignOPTAttack, GeoDA
# Import other attacks as needed

class AttackExecutor:
    """
    The AttackExecutor class is responsible for initializing and executing adversarial attacks on a given model.
    """
    def __init__(self, attack_config, model):
        """
        Initializes the AttackExecutor with an attack configuration and a model.
        
        :param attack_config: A dictionary containing the configuration for the attack to be executed.
                              This includes the name of the attack and any necessary parameters.
        :param model: The machine learning model to be attacked.
        """
        self.attack_config = attack_config
        self.model = model
        self.attack = self.initialize_attack()

    def initialize_attack(self):
        """
        Initializes the attack based on the configuration provided during object creation.
        
        :return: An instance of the specified attack.
        """
        attack_name = self.attack_config['name']
        if attack_name == 'ZooAttack':
            max_iter = self.attack_config.get('max_iter')
            learning_rate = self.attack_config.get('learning_rate')
            binary_search_steps = self.attack_config.get('binary_search_steps')
            use_resize = self.attack_config.get('use_resize')
            variable_h = self.attack_config.get('variable_h')

            attack = ZooAttack(classifier=self.model, max_iter=max_iter,learning_rate=learning_rate,
                               binary_search_steps=binary_search_steps,use_resize=use_resize,variable_h=variable_h)
            return attack
        
        elif attack_name == 'HopSkipJump':
            max_iter = self.attack_config.get('max_iter')
            max_eval = self.attack_config.get('max_eval')
            init_eval = self.attack_config.get('init_eval')
            init_size = self.attack_config.get('init_size')
            norm = self.attack_config.get('norm')

            attack = HopSkipJump(classifier=self.model,max_iter=max_iter,
                                 max_eval=max_eval,init_eval=init_eval,init_size=init_size,norm=norm)
            return attack
        
        elif attack_name == 'SignOPTAttack':
            targeted = self.attack_config.get('targeted')
            epsilon = self.attack_config.get('epsilon')

            attack = SignOPTAttack(estimator=self.model,targeted=targeted,epsilon=epsilon)
            return attack
        
        elif attack_name == 'GeoDA':
            max_iter = self.attack_config.get('max_iter')
            sigma = self.attack_config.get('sigma')
            bin_search_tol = self.attack_config.get('bin_search_tol')
            norm = self.attack_config.get('norm')

            attack = GeoDA(estimator=self.model,max_iter=max_iter,sigma=sigma,bin_search_tol=bin_search_tol,norm=norm)
            return attack
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

    def execute_attack(self, x):
        """
        Executes the initialized attack on the provided input data.
        
        :param x: The input data to be attacked.
        :return: The adversarially perturbed input data.
        """
        x_adv = self.attack.generate(x=x)
        return x_adv
