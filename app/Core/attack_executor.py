from art.attacks.evasion import ZooAttack
# Import other attacks as needed

class AttackExecutor:
    def __init__(self, attack_config, model, x_test, y_test):
        self.attack_config = attack_config
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.attack = self.initialize_attack()

    def initialize_attack(self):
        attack_name = self.attack_config['name']
        if attack_name == 'ZooAttack':
            # Initialize ZooAttack with parameters from self.attack_config
            # For demonstration, let's assume 'parameter_x' is a config parameter for ZooAttack
            max_iter = self.attack_config.get('max_iter')
            learning_rate = self.attack_config.get('learning_rate')
            binary_search_steps = self.attack_config.get('binary_search_steps')
            use_resize = self.attack_config.get('use_resize')
            variable_h = self.attack_config.get('variable_h')

            attack = ZooAttack(classifier=self.model, max_iter=max_iter,learning_rate=learning_rate,
                               binary_search_steps=binary_search_steps,use_resize=use_resize,variable_h=variable_h)
            return attack
        # Add other attacks here as elif branches
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

    def execute_attack(self):
        x_adv = self.attack.generate(x=self.x_test)
        return x_adv
