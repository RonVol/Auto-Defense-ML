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
        "type:":"black-box",
        "max_iter": 10,
        "learning_rate": 1e-1,
        "binary_search_steps": 20,
        "use_resize": False,
        "variable_h":0.05,
        "applicable_to": ["XGBoost"]
    },
    "HopSkipJump": {
        "name": "HopSkipJump",
        "type:":"black-box",
        "max_iter": 50,
        "max_eval": 10000,
        "init_eval": 100,
        "init_size": 100,
        "norm":2,
        "applicable_to": ["XGBoost"]
    },
    "SignOPTAttack": { # TODO : self.clip_min not defined, bug in ART ? 
        "name": "SignOPTAttack",
        "type:":"black-box",
        "targeted": False,
        "epsilon": 0.001,
        "applicable_to": ["XGBoost"]
    },
    "GeoDA": {
        "name": "GeoDA", # 'XGBoostClassifier' object has no attribute 'channels_first'
        "type:":"black-box",
        "max_iter": 4000,
        "sigma": 0.0002,
        "bin_search_tol": 0.1,
        "norm": 2,
        "applicable_to": ["XGBoost"]
    },
}

supported_defenses = {
    "SpatialSmoothing": {
        "name": "SpatialSmoothing",
        "window_size": 3,
        "apply_fit": False,
        "apply_predict": True,
        "applicable_to": ["XGBoost"]
    },
    "JpegCompression": { # Negative values in input `x` detected. The JPEG compression defence requires unnormalized input.
        "name": "JpegCompression",
        "clip_values": (0,1),
        "quality": 50,
        "apply_fit": False,
        "apply_predict": True,
        "applicable_to": ["XGBoost"]
    },
    "ThermometerEncoding": { # need to see what happends to the shape, Found input variables with inconsistent numbers of samples: [10, 100]
        "name": "ThermometerEncoding",
        "clip_values": (0,1),
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

