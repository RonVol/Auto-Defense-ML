import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

from app.config import supported_libraries, supported_attacks, supported_defenses
from utils.mnist_xgboost import get_model_path,get_x_test_path,get_y_test_path # for simulated input

class Main_UI():

    def __init__(self):
        print("UI init")
        self.run_ui()

    def analyze(self, sender, app_data):
        text_value = dpg.get_value(self.input_text_id)
        checkbox_value = dpg.get_value(self.type)
        dropdown1_value = dpg.get_value(self.attacks_drop)
        dropdown2_value = dpg.get_value(self.defenses_drop)
        self.extracted_data = {
            "Path to model": text_value,
            "BlackBox": checkbox_value,
            "Chosen Attack": dropdown1_value,
            "Chosen Defence": dropdown2_value
        }
        print(self.extracted_data)


    def button_callback(sender, app_data):
        print("Button", app_data, "pressed")

    def checkbox_callback(sender, app_data):
        print("Checkbox state:", dpg.get_value(sender))

    def dropdown_callback(sender, app_data):
        print("Dropdown value changed to:", app_data)

    def clear_items(self, sender, app_data):
        dpg.set_value(self.input_text_id, "")
        dpg.set_value(self.type, False)
        dpg.set_value(self.attacks_drop, "")
        dpg.set_value(self.defenses_drop, "")
        print("All items cleared.")

    def run_ui(self):
        attacks = list(supported_attacks)
        defenses = list(supported_defenses)
        dpg.create_context()
        dpg.create_viewport(title='Dear PyGui Example', width=500, height=150)
        with dpg.window(label="GUI Window", width=500, height=300):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Model", callback=self.button_callback, user_data=1)
                dpg.add_text("Model's Path:")
                self.input_text_id = dpg.add_input_text(label="", width=200)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Choose Attacks", callback=self.button_callback, user_data=2)
                self.attacks_drop = dpg.add_combo(label="", items=attacks, callback=self.dropdown_callback)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Choose Defences", callback=self.button_callback, user_data=3)
                self.defenses_drop = dpg.add_combo(label="", items=defenses, callback=self.dropdown_callback)
            self.type = dpg.add_checkbox(label="Blackbox", callback=self.checkbox_callback)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Analyze", callback=self.analyze, user_data=4)
                dpg.add_button(label="Clear", callback=self.clear_items, user_data=5)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self):
        dpg.start_dearpygui()


    def get_simulated_user_input(self):
        chosen_model_file_path = get_model_path()
        chosen_model_library = supported_libraries['XGBoost']
        chosen_attacks = [supported_attacks[self.extracted_data["Chosen Attack"]]] # list of chosen attacks
        chosen_defenses = [supported_defenses[self.extracted_data["Chosen Defence"]]]
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