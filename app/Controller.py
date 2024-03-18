from app.Core.main_core import Main_Core
from app.UI.main_ui import Main_UI
import app.config as config
from app.data_loader import DataLoader
from app.Reports.report_generator import Report_Generator

class Controller:
    def __init__(self, core : Main_Core, ui : Main_UI):
        self.core = core
        self.ui = ui
        self.ui.set_controller(self)
        self.dataloader = None

    def handle_load(self, model_path, x_path, y_path, selected_library):
        print("Model Path:", model_path)
        print("Selected Library:", selected_library)
        print("X Path:", x_path)
        print("Y Path:", y_path)
        dataloader, is_success, message = self.create_dataloader(model_path, selected_library, x_path, y_path)
        print(message)
        self.dataloader = dataloader
        return is_success
    
    def handle_configuration(self, selected_attacks, selected_defenses, chosen_run):
        print("in handle_configuration")
        if chosen_run == 1: # default parameters or manually configured
            print("in handle_configuration chosen_run == 1")
            self.start_main_pipeline(selected_attacks, selected_defenses)
        elif chosen_run == 2: # optimize
            self.start_main_pipeline(selected_attacks, selected_defenses)


    def run_ui(self):
        self.ui.run()
    
    def start_main_pipeline(self,attacks, defenses):
        print("in start_main_pipelin")
        self.core.dataloader = self.dataloader
        self.ui.update_progress("Performing benign evaluation...")
        clean_metrics = self.core.perform_benign_evaluation()
        self.ui.update_progress("perform_attacks...")
        metrics_att, adv_examples = self.core.perform_attacks(attacks)
        self.ui.update_progress("perform_defenses...")
        metrics_deff, defended_examples = self.core.perform_defenses(defenses)
        self.ui.update_progress("perform_defenses_on_attacks...")
        metrics_att_def, adv_defended_examples = self.core.perform_defenses_on_attacks(defenses,adv_examples)
        self.ui.update_progress("DONE!")
        print(40*"-")
        print(clean_metrics)
        print(40*"-")
        print(metrics_att)
        print(40*"-")
        print(metrics_deff)
        print(40*"-")
        print(metrics_att_def)


    def create_dataloader(self,model_fpath, model_library, x_test_fpath, y_test_fpath):
        dataloader = DataLoader()
        message = "Data Loading Status: "
        if not dataloader.load_model(model_library, model_fpath):
            message = message + " Failed to load the model: " + model_fpath
            return dataloader, False, message

        if not dataloader.load_test(x_test_fpath,y_test_fpath):
            message = message + " Failed to load test data."
            return dataloader, False, message

        message = message + " Success"
        return dataloader, True, message

