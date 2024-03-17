from app.Core.main_core import Main_Core
from app.UI.main_ui import Main_UI
import app.config as config
from app.data_loader import DataLoader
from app.Reports.report_generator import Report_Generator

class Controller:
    def __init__(self, core : Main_Core, ui : Main_UI):
        self.core = core
        self.ui = ui

    def run_ui(self):
        #self.ui.run_ui()

        #for testing without ui
        self.validate_user_input()
        metrics = self.start_main_pipeline()
        report = Report_Generator(metrics)
        report.build_json()


    def validate_user_input(self):
        inp = self.ui.get_simulated_user_input()# for now simulated, later change it.
        #if config.validate_user_input(inp): should perform validation to the schema
        self.user_input = inp
    
    def start_main_pipeline(self):
        dataloader = self.create_dataloader()
        parameters = self.get_user_input_parameters()
        attacks = self.get_user_chosen_attacks()
        defenses = self.get_user_chosen_defenses()
        return self.core.main_loop(dataloader,parameters,attacks,defenses)
        #print(dataloader.y_test[0])

    def get_user_chosen_attacks(self):
        chosen_attacks = self.user_input['chosen_attacks']
        return chosen_attacks
    
    def get_user_chosen_defenses(self):
        chosen_defenses = self.user_input['chosen_defenses']
        return chosen_defenses

    def get_user_input_parameters(self):
        nb_classes = self.user_input['nb_classes']
        nb_features = self.user_input['nb_features']   
        return {'nb_classes':nb_classes, 'nb_features':nb_features}

    def create_dataloader(self):
        dataloader = DataLoader()
        model_fpath = self.user_input['model_path']
        x_test_fpath = self.user_input['x_test_path']
        y_test_fpath = self.user_input['y_test_path']
        model_library = self.user_input['model_library']

        if not dataloader.load_model(model_library, model_fpath):
            # pass to ui that failed to load model
            pass

        if not dataloader.load_test(x_test_fpath,y_test_fpath):
            # pass to ui that failed to load test
            pass

        return dataloader

