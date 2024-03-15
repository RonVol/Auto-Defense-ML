from app.Core.main_core import Main_Core
from app.UI.main_ui import Main_UI
from app.controller import Controller

if __name__ == "__main__":
    # create controller for interaction between core and gui
    core = Main_Core()
    ui = Main_UI()
    controller = Controller(core=core,ui=ui)
    controller.run_ui()