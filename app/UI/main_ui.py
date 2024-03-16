import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

from app.config import supported_libraries, supported_attacks, supported_defenses
from utils.mnist_xgboost import get_model_path,get_x_test_path,get_y_test_path # for simulated input

class Main_UI():

    def __init__(self):
        print("UI init")

    def run_ui(self):
        dpg.create_context()
        dpg.create_viewport(title='A Title', width=600, height=600)

        demo.show_demo()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def get_simulated_user_input(self):
        chosen_model_file_path = get_model_path()
        chosen_model_library = supported_libraries['XGBoost']
        chosen_attacks = [supported_attacks['ZooAttack']] # list of chosen attacks
        chosen_defenses = [supported_defenses['SpatialSmoothing']]
        chosen_x_test = get_x_test_path()
        chosen_y_test = get_y_test_path()
        nb_classes = 10
        nb_features = 28*28
        return {"model_path":chosen_model_file_path,
                "model_library":chosen_model_library,
                "chosen_attacks":chosen_attacks,
                "chosen_defenses":chosen_defenses,
                "x_test_path":chosen_x_test,
                "y_test_path":chosen_y_test,
                "nb_classes":nb_classes,
                "nb_features":nb_features}