from jsonschema import validate
import jsonschema

supported_libraries = {
    "XGBoost": {
        "name": "XGBoost",
        "requirements": ["model", "nb_features", "nb_classes"]
    },
    "scikit-learn": {
        "name": "scikit-learn",
        "requirements": ["model", "nb_features", "nb_classes"]
    },
}

supported_attacks = {
    "ZooAttack": {
        "name": "ZooAttack",
        "type":"black-box",
        "targeted": False,
        "confidence":0.0,
        "max_iter": 20,
        "learning_rate": 1e-1,
        "initial_const":1e-3,
        "binary_search_steps": 10,
        "batch_size":1,
        "use_importance": False,
        "nb_parallel": 1,
        "abort_early": True,
        "use_resize": False,
        "variable_h":0.2,
        "applicable_to": ["XGBoost", "scikit-learn"]
    },
    "HopSkipJump": {
        "name": "HopSkipJump",
        "type":"black-box",
        "targeted": False,
        "batch_size":1,
        "max_iter": 100,
        "max_eval": 100,
        "init_eval": 100,
        "init_size": 100,
        "norm":2,
        "applicable_to": ["XGBoost", "scikit-learn"]
    },
    "SignOPTAttack": { # self.clip_min not defined, bug in ART ? temporary fix by defining it manually in executor
        "name": "SignOPTAttack",
        "type":"black-box",
        "targeted": False,
        "epsilon": 0.001,
        "num_trial": 10,
        "max_iter": 10,
        "query_limit": 20000,
        "k": 200,
        "alpha": 0.2,
        "beta": 0.001,
        "batch_size": 1,
        "applicable_to": ["XGBoost", "scikit-learn"]
    },
    "BoundaryAttack": {
        "name": "BoundaryAttack",
        "type":"black-box",
        "targeted": False,
        "batch_size": 1,
        "delta":0.01,
        "epsilon":0.01,
        "step_adapt":0.667,
        "max_iter": 100,
        "num_trial": 25,
        "sample_size": 20,
        "init_size": 100,
        "min_epsilon":0.0,
        "applicable_to": ["XGBoost", "scikit-learn"]
    },
    "Papernot_DT_Attack": {
        "name": "Papernot_DT_Attack",
        "type":"black-box",
        "offset": 0.001,
        "applicable_to": ["scikit-learn"]
    },
    "CubeAttack": {
        "name": "CubeAttack",
        "type":"black-box",
        "eps": 0.1,
        "n_trials":100,
        "p":0.5,
        "independent_delta":False,
        "applicable_to": ["XGBoost"]
    },
    "SamplingAttack": {
        "name": "SamplingAttack",
        "type":"black-box",
        "eps": 0.1,
        "n_trials":100,
        "applicable_to": ["XGBoost","scikit-learn"]
    },
}

supported_defenses = {
    "FeatureSqueezing": {
        "name": "FeatureSqueezing",
        "defense_type" : "preprocessor",
        "bit_depth": 3,
        "apply_fit": False,
        "apply_predict": True,
        "applicable_to": ["XGBoost", "scikit-learn"]
    },
    "ClassLabels": {
        "name": "ClassLabels",
        "defense_type" : "postprocessor",
        "apply_fit": False,
        "apply_predict": True,
        "applicable_to": ["XGBoost", "scikit-learn"]
    },
    "TTTS": {
        "name": "TTTS",
        "defense_type" : "new_classifier",
        "prob_type": "depth",
        "n_simulations": 10,
        "applicable_to": ["scikit-learn"]
    },
}

