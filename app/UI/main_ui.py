import dearpygui.dearpygui as dpg
from app.config import supported_libraries, supported_attacks, supported_defenses

class Main_UI:
    def __init__(self):
        self.controller = None
        self.def_model_path = "models/iris_xgboost.model"
        self.def_x_path = "models/iris_xgboost_x_test.npy"
        self.def_y_path = "models/iris_xgboost_y_test.npy"
        self.setup_ui()

    def set_controller(self, controller):
        self.controller = controller

    def setup_ui(self):
        dpg.create_context()
        with dpg.window(label="Main Window", tag="Main_Window", width=1280, height=720, no_move=True):
                with dpg.group(horizontal=False):  # Organize vertically
                    dpg.add_text("Enter File Paths")
                with dpg.group(horizontal=True):  # Label on the left
                    dpg.add_text("Model File Path: ")
                    self.model_path_id = dpg.add_input_text(default_value=self.def_model_path, width=-1)
                            # Creating the dropdown for library selection
                library_names = [lib["name"] for lib in supported_libraries.values()]
                with dpg.group(horizontal=True):
                    dpg.add_text("Select Library: ")
                    self.library_id = dpg.add_combo(library_names, default_value=library_names[0] if library_names else "")

                with dpg.group(horizontal=True):  # Label on the left
                    dpg.add_text("X File Path: ")
                    self.x_path_id = dpg.add_input_text(default_value=self.def_x_path, width=-1)
                with dpg.group(horizontal=True):  # Label on the left
                    dpg.add_text("Y File Path: ")
                    self.y_path_id = dpg.add_input_text(default_value=self.def_y_path, width=-1)
                
                dpg.add_button(label="Load", callback=self.load_files)

    def setup_attack_defense_window(self):
        with dpg.window(label="Select Attacks and Defenses",tag="Select Attacks and Defenses", width=1280, height=720, no_move=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Select Attacks")
                for attack in supported_attacks:
                    dpg.add_checkbox(label=attack, tag=f"attack_{attack}")

                dpg.add_button(label="Select All Attacks", callback=lambda: self.select_all(True, supported_attacks, prefix="attack_"))
                dpg.add_button(label="Select None Attacks", callback=lambda: self.select_all(False, supported_attacks, prefix="attack_"))

                dpg.add_separator()

                dpg.add_text("Select Defenses")
                for defense in supported_defenses:
                    dpg.add_checkbox(label=defense, tag=f"defense_{defense}")

                dpg.add_button(label="Select All Defenses", callback=lambda: self.select_all(True, supported_defenses, prefix="defense_"))
                dpg.add_button(label="Select None Defenses", callback=lambda: self.select_all(False, supported_defenses, prefix="defense_"))

            dpg.add_separator()
            with dpg.group(horizontal=False):
                dpg.add_text("Parameter Configuration Options")
                with dpg.group():
                    # Using a radio button for exclusive selection
                    self.param_config_option = dpg.add_radio_button(["Run on Default Parameters", 
                                                                    "Optimize Attack and Defense Parameters", 
                                                                    "Configure Manually Attack and Defense Parameters"], 
                                                                    horizontal=False)
                dpg.add_button(label="Proceed", callback=self.on_proceed_with_selection)

    def setup_parameter_configuration_window(self, selected_attacks, selected_defenses):
        with dpg.window(label="Configure Parameters",tag="Configure Parameters", width=1280, height=720, no_move=True):
            # Configure Attack Parameters
            with dpg.group(horizontal=False):
                dpg.add_text("Configure Attack Parameters")
                for attack_config in selected_attacks:
                    dpg.add_text(attack_config["name"])  # Display the name of the attack
                    for param, value in attack_config.items():
                        # Exclude non-parameter items
                        if param not in ["name", "type", "applicable_to"]:
                            with dpg.group(horizontal=True):  # Grouping label and input horizontally
                                dpg.add_text(f"{param}: ")  # Fixed width for the label
                                # Shorter input field with specific width
                                if isinstance(value,bool):
                                        dpg.add_checkbox(default_value=value, tag=f"{attack_config['name']}_{param}")
                                else:
                                    dpg.add_input_text(default_value=str(value), width=200, tag=f"{attack_config['name']}_{param}")

            dpg.add_separator()

            # Configure Defense Parameters
            with dpg.group(horizontal=False):
                dpg.add_text("Configure Defense Parameters")
                for defense_config in selected_defenses:
                    dpg.add_text(defense_config["name"])  # Display the name of the defense
                    for param, value in defense_config.items():
                        # Exclude non-parameter items
                        if param not in ["name", "applicable_to"]:
                            with dpg.group(horizontal=True):  # Grouping label and input horizontally
                                dpg.add_text(f"{param}: ")  # Fixed width for the label
                                # Shorter input field with specific width
                                if isinstance(value,bool):
                                    dpg.add_checkbox(default_value=value, tag=f"{defense_config['name']}_{param}")
                                else:
                                    dpg.add_input_text(default_value=str(value), width=200, tag=f"{defense_config['name']}_{param}")

            dpg.add_separator()
            dpg.add_button(label="Begin Run", callback=self.on_proceed_with_selection_manual_config)


    def on_proceed_with_selection(self):
        selected_option = dpg.get_value(self.param_config_option)
        self.selected_attacks = [supported_attacks[attack] for attack in supported_attacks if dpg.get_value(f"attack_{attack}")]
        self.selected_defenses = [supported_defenses[defense] for defense in supported_defenses if dpg.get_value(f"defense_{defense}")]
        
        if selected_option == "Configure Manually Attack and Defense Parameters":
            dpg.hide_item("Select Attacks and Defenses")  # Assuming this is the tag for the attack/defense selection window
            self.setup_parameter_configuration_window(self.selected_attacks, self.selected_defenses)
        else:
            # Pass the selections along with the chosen option (1 for default, 2 for optimize) back to the controller
            if self.controller:
                dpg.hide_item("Select Attacks and Defenses")
                self.show_progress_window()
                self.controller.handle_configuration(self.selected_attacks, self.selected_defenses, 1 if selected_option == "Run on Default Parameters" else 2)
            else:
                print("Controller not set!")

    def on_proceed_with_selection_manual_config(self):
        # Gather updated configurations for attacks
        updated_attacks_config = []
        for attack_config in self.selected_attacks:
            # Initially copy essential attributes
            updated_config = {key: attack_config[key] for key in ["name", "type", "applicable_to"]}
            # Update each parameter with the new value from the input field
            for param, value in attack_config.items():
                if param not in updated_config:  # Exclude already copied essential attributes
                    input_value = dpg.get_value(f"{attack_config['name']}_{param}")
                    # Convert the input value back to the appropriate type (int, float, bool)
                    original_type = type(value)
                    updated_config[param] = original_type(input_value) if input_value else value  # Fallback to original value if input is empty
            updated_attacks_config.append(updated_config)
        
        # Similarly, gather updated configurations for defenses
        updated_defenses_config = []
        for defense_config in self.selected_defenses:
            # Initially copy essential attributes
            updated_config = {key: defense_config[key] for key in ["name", "applicable_to"]}
            for param, value in defense_config.items():
                if param not in updated_config:  # Exclude already copied essential attributes
                    input_value = dpg.get_value(f"{defense_config['name']}_{param}")
                    original_type = type(value)
                    updated_config[param] = original_type(input_value) if input_value else value  # Fallback to original value if input is empty
            updated_defenses_config.append(updated_config)
        
        # Now call the controller's handle_configuration method with these updated configurations
        if self.controller:
            dpg.hide_item("Configure Parameters")
            self.show_progress_window()
            self.controller.handle_configuration(updated_attacks_config, updated_defenses_config, 1)  # Adjusted to denote manual configuration

        else:
            print("Controller not set!")

    def show_progress_window(self):
        with dpg.window(label="Pipeline Progress", tag="progress_window", width=1280, height=720, no_move=True):
            # Adding spacers for vertical alignment
            dpg.add_spacer(height=300)  # Adjust height for better vertical alignment
            
            # Text centering through layout adjustment
            with dpg.group(horizontal=True, width=980/2):
                dpg.add_spacer()  # Auto-adjusts to center the following widget
                dpg.add_text("Starting pipeline...", tag="progress_text")
                dpg.add_spacer()  # Ensures the text widget is centered
            
            dpg.add_spacer(height=10)  # Adjust height for better vertical alignment

            
    def update_progress(self, message):
        if dpg.does_item_exist("progress_text"):
            dpg.set_value("progress_text",dpg.get_value("progress_text")+'\n'+ message)
        else:
            print("Progress window is not open or progress text does not exist.")


    def select_all(self, select, items, prefix):
        for item in items:
            dpg.set_value(f"{prefix}{item}", select)
    
    def load_files(self):
        model_path = dpg.get_value(self.model_path_id)
        x_path = dpg.get_value(self.x_path_id)
        y_path = dpg.get_value(self.y_path_id)
        selected_library = supported_libraries[dpg.get_value(self.library_id)]

        if self.controller:
            success = self.controller.handle_load(model_path, x_path, y_path, selected_library)
            if success:
                dpg.hide_item("Main_Window")  # Assuming "Main Window" is the tag of the main window
                self.setup_attack_defense_window()
            else:
                print("Load unsuccessful.")
        else:
            print("Controller not set!")
    
    def run(self):
        dpg.create_viewport(title='Custom Title', width=1280, height=720)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
