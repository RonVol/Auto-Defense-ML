import dearpygui.dearpygui as dpg
import dearpygui.demo as demo


from config import supported_libraries, supported_attacks
from utils.mnist_xgboost import get_model_path,get_x_test_path,get_y_test_path # for simulated input
class UI():

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

    def get_simulated_user_input():
        chosen_model_file_path = get_model_path()
        chosen_model_library = supported_libraries['XGBoost']
        chosen_attacks = [supported_attacks['FGSM']] # list of chosen attacks
        chosen_x_test = get_x_test_path()
        chosen_y_test = get_y_test_path()
        return {"model_path":chosen_model_file_path,
                "model_library": chosen_model_library,
                "chosen_attacks":chosen_attacks,
                "x_test":chosen_x_test,
                "y_test":chosen_y_test}