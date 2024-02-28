from .UI.main_ui import UI
from .Core.main_core import Main_Core
from .config import supported_libraries, supported_attacks
from .Controller import Controller
# imports for ui simulation
from utils.mnist_xgboost import get_model_path,get_x_test_path,get_y_test_path


if __name__ == "__main__":

    # create controller for interaction between core and gui
    core = Main_Core()
    ui = UI()
    controller = Controller(core=core,ui=ui)
    controller.run_ui()

    
