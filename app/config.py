supported_libraries = {
    "XGBoost": {
        "description": "Gradient boosting library",
        "supported_attacks": ["FGSM"],
        "supported_defenses": []
    },
}

supported_attacks = {
    "FGSM": {
        "description": "Fast Gradient Sign Method",
        "applicable_to": ["XGBoost"]
    },
}

# supported_defenses = {
#     "GaussianNoise": {
#         "description": "Adding Gaussian noise to inputs",
#         "applicable_to": ["XGBoost"]
#     },
# }