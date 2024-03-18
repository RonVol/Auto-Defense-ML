from jsonschema import validate
import jsonschema

supported_libraries = {
    "XGBoost": {
        "name": "XGBoost",
        "requirements": ["model", "nb_features", "nb_classes"],
        "supported_attacks": ["FGSM"],
        "supported_defenses": []
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
        "applicable_to": ["XGBoost"]
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
        "applicable_to": ["XGBoost"]
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
        "applicable_to": ["XGBoost"]
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
        "applicable_to": ["XGBoost"]
    },
}

supported_defenses = {
    "FeatureSqueezing": {
        "name": "FeatureSqueezing",
        "bit_depth": 3,
        "apply_fit": False,
        "apply_predict": True,
        "applicable_to": ["XGBoost"]
    },
}

def get_user_input_json_schema(model_library, supported_libraries, supported_attacks, supported_defenses):
    # Initialize the basic schema structure
    schema = {
        "type": "object",
        "properties": {
            "model_path": {"type": "string"},
            "model_library": {"type": "string", "enum": list(supported_libraries.keys())},
            "chosen_attacks": {
                "type": "array",
                "items": {"type": "string", "enum": list(supported_attacks.keys())},
                "minItems": 1
            },
            "chosen_defenses": {
                "type": "array",
                "items": {"type": "string", "enum": list(supported_defenses.keys())},
                "minItems": 0
            },
            "x_test_path": {"type": "string"},
            "y_test_path": {"type": "string"},
        },
        "required": ["model_path", "model_library", "chosen_attacks", "x_test_path", "y_test_path"],
        "additionalProperties": False,
    }
    
    # Include library-specific requirements for the selected library
    library_requirements = supported_libraries[model_library].get("requirements", [])
    for req in library_requirements:
        # Here you might want to customize the type or validation based on the requirement
        schema["properties"][req] = {"type": "string"}
        if req not in schema["required"]:
            schema["required"].append(req)

    return schema

def validate_model_library(input_data):
    library_schema = {
        "type": "object",
        "properties": {
            "model_library": {"type": "string", "enum": list(supported_libraries.keys())},
        },
        "required": ["model_library"],
        "additionalProperties": False,
    }
    try:
        validate(instance=input_data, schema=library_schema)
        return True, input_data["model_library"]  # Return True and the library name if valid
    except jsonschema.exceptions.ValidationError as e:
        print(f"Validation error: {e.message}")
        return False, None

def validate_user_input(input_data):
    # Validate model_library first
    is_library_valid, model_library = validate_model_library(input_data)
    if not is_library_valid:
        return False
    
    # Generate and validate against the full schema with requirements for the selected library
    schema = get_user_input_json_schema(model_library, supported_libraries, supported_attacks, supported_defenses)
    try:
        validate(instance=input_data, schema=schema)
        print("Validation passed.")
        return True
    except jsonschema.exceptions.ValidationError as e:
        print(f"Validation error: {e.message}")
        return False

