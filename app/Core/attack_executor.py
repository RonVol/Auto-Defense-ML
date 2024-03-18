from art.attacks.evasion import ZooAttack, HopSkipJump, SignOPTAttack, BoundaryAttack
import numpy as np
# Import other attacks as needed

class AttackExecutor:
    """
    The AttackExecutor class is responsible for initializing and executing adversarial attacks on a given model.
    """
    def __init__(self, attack_config, model, clip_values):
        """
        Initializes the AttackExecutor with an attack configuration and a model.
        
        :param attack_config: A dictionary containing the configuration for the attack to be executed.
                              This includes the name of the attack and any necessary parameters.
        :param model: The machine learning model to be attacked.
        """
        self.attack_config = attack_config
        self.model = model
        self.clip_values = clip_values
        self.attack = self.initialize_attack()

    def initialize_attack(self):
        """
        Initializes the attack based on the configuration provided during object creation.
        
        :return: An instance of the specified attack.
        """
        attack_name = self.attack_config['name']
        if attack_name == 'ZooAttack':
            max_iter = self.attack_config.get('max_iter')
            targeted = self.attack_config.get('targeted')
            confidence = self.attack_config.get('confidence')
            initial_const = self.attack_config.get('initial_const')
            batch_size = self.attack_config.get('batch_size')
            use_importance = self.attack_config.get('use_importance')
            nb_parallel = self.attack_config.get('nb_parallel')
            abort_early = self.attack_config.get('abort_early')
            learning_rate = self.attack_config.get('learning_rate')
            binary_search_steps = self.attack_config.get('binary_search_steps')
            use_resize = self.attack_config.get('use_resize')
            variable_h = self.attack_config.get('variable_h')

            attack = ZooAttack(classifier=self.model, max_iter=max_iter,learning_rate=learning_rate,
                               binary_search_steps=binary_search_steps,use_resize=use_resize,variable_h=variable_h,confidence=confidence,initial_const=initial_const,
                               batch_size=batch_size,use_importance=use_importance,nb_parallel=nb_parallel,abort_early=abort_early)
            return attack
        
        elif attack_name == 'HopSkipJump':
            max_iter = self.attack_config.get('max_iter')
            max_eval = self.attack_config.get('max_eval')
            init_eval = self.attack_config.get('init_eval')
            init_size = self.attack_config.get('init_size')
            #norm = self.attack_config.get('norm')
            norm = np.inf
            batch_size = self.attack_config.get('norm')
            targeted = self.attack_config.get('targeted')
            attack = HopSkipJump(classifier=self.model,max_iter=max_iter,
                                 max_eval=max_eval,init_eval=init_eval,init_size=init_size,norm=norm,batch_size=batch_size,targeted=targeted)
            return attack
        
        elif attack_name == 'SignOPTAttack':
            targeted = self.attack_config.get('targeted')
            epsilon = self.attack_config.get('epsilon')
            max_iter = self.attack_config.get('max_iter')
            num_trial = self.attack_config.get('num_trial')
            query_limit = self.attack_config.get('query_limit')
            k = self.attack_config.get('k')
            alpha = self.attack_config.get('alpha')
            beta = self.attack_config.get('beta')
            batch_size = self.attack_config.get('batch_size')

            attack = SignOPTAttack(estimator=self.model,targeted=targeted,epsilon=epsilon,max_iter=max_iter,
                                   num_trial=num_trial,query_limit=query_limit,k=k,alpha=alpha,beta=beta,verbose=True)
            attack.clip_min = self.clip_values[0]
            attack.clip_max = self.clip_values[1]
            return attack
        
        elif attack_name == 'BoundaryAttack':
            max_iter = self.attack_config.get('max_iter')
            batch_size = self.attack_config.get('batch_size')
            targeted = self.attack_config.get('targeted')
            delta = self.attack_config.get('delta')
            epsilon = self.attack_config.get('epsilon')
            step_adapt = self.attack_config.get('step_adapt')
            num_trial = self.attack_config.get('num_trial')
            sample_size = self.attack_config.get('sample_size')
            init_size = self.attack_config.get('init_size')
            min_epsilon = self.attack_config.get('min_epsilon')


            attack = BoundaryAttack(estimator=self.model, max_iter=max_iter, batch_size=batch_size,
                                    epsilon=epsilon,targeted=targeted,delta=delta,step_adapt=step_adapt,
                                    num_trial=num_trial,sample_size=sample_size,init_size=init_size,min_epsilon=min_epsilon)
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
