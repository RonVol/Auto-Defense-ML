import dearpygui.dearpygui as dpg
from app.config import supported_libraries, supported_attacks, supported_defenses
import os
from datetime import datetime

class Main_UI:
    def __init__(self):
        self.controller = None
        self.def_model_path = "models/iris_xgboost.model"
        self.def_x_path = "models/iris_xgboost_x_test.npy"
        self.def_y_path = "models/iris_xgboost_y_test.npy"
        self.from_history = False
        self.setup_ui()
 

    def set_controller(self, controller):
        self.controller = controller

    def load_callback(self, sender, app_data, user_data):
        self.model_dialog(user_data)
    
    def load_callback_run(self, sender, app_data, user_data):
        self.history_dialog(user_data)

    def load_callback_xy(self, sender, app_data, user_data):
        self.xy_dialog(user_data)
       
    
    def file_selected_callback(self, sender, app_data, user_data):
        selected_file = app_data["file_path_name"]
        target_input_id = user_data
        print(user_data)
        dpg.set_value(target_input_id, selected_file)
        if dpg.does_item_exist("model_dialog_id"):
            dpg.delete_item("model_dialog_id")
        if dpg.does_item_exist("history_dialog_id"):
            dpg.delete_item("history_dialog_id")
        if dpg.does_item_exist("xy_dialog_id"):
            dpg.delete_item("xy_dialog_id")
        dpg.show_item("Main_Window")

    def history_dialog(self, data):
        if not dpg.does_item_exist("history_dialog_id"):
            with dpg.file_dialog(directory_selector=False, show=False, callback=self.file_selected_callback, cancel_callback=self.cancel_callback, user_data=data,  id="history_dialog_id", height=600, width=600):
                dpg.add_file_extension(".txt", color=(150, 255, 150, 255))
        dpg.show_item("history_dialog_id")
        dpg.hide_item("Main_Window")

    def model_dialog(self, data):
        if not dpg.does_item_exist("model_dialog_id"):
            with dpg.file_dialog(directory_selector=False, show=False, callback=self.file_selected_callback, cancel_callback=self.cancel_callback, user_data=data,  id="model_dialog_id", height=600, width=600):
                dpg.add_file_extension(".model", color=(150, 255, 150, 255))
        dpg.show_item("model_dialog_id")
        dpg.hide_item("Main_Window")

    def xy_dialog(self, data):
        if not dpg.does_item_exist("xy_dialog_id"):
            with dpg.file_dialog(directory_selector=False, show=False, callback=self.file_selected_callback, cancel_callback=self.cancel_callback, user_data=data, id="xy_dialog_id", height=600, width=600):
                dpg.add_file_extension(".npy", color=(150, 255, 150, 255))
        dpg.show_item("xy_dialog_id")
        dpg.hide_item("Main_Window")
    
    def cancel_callback(self, sender, app_data, user_data):
        dpg.show_item("Main_Window")
    
    def setup_ui(self):
        dpg.create_context()
        with dpg.window(label="Main Window", tag="Main_Window", width=1280, height=720, no_move=True):
                with dpg.group(horizontal=False):  # Organize vertically
                    dpg.add_text("Enter File Paths")
                with dpg.group(horizontal=True):  # Label on the left
                    dpg.add_text("Model File Path: ")
                    self.model_path_id = dpg.add_input_text(default_value=self.def_model_path, tag="model_path",width=250)
                    dpg.add_button(label="browse", callback=self.load_callback, user_data=self.model_path_id)
                            # Creating the dropdown for library selection
                library_names = [lib["name"] for lib in supported_libraries.values()]
                with dpg.group(horizontal=True):
                    dpg.add_text("Select Library: ")
                    self.library_id = dpg.add_combo(library_names,tag="model_library",width=160, default_value=library_names[0] if library_names else "")

                with dpg.group(horizontal=True):  # Label on the left
                    dpg.add_text("X File Path: ")
                    self.x_path_id = dpg.add_input_text(default_value=self.def_x_path, tag="x_path",width=320)
                    dpg.add_button(label="browse", callback=self.load_callback_xy, user_data=self.x_path_id)
                with dpg.group(horizontal=True):  # Label on the left
                    dpg.add_text("Y File Path: ")
                    self.y_path_id = dpg.add_input_text(default_value=self.def_y_path, tag="y_path",width=320)
                    dpg.add_button(label="browse", callback=self.load_callback_xy, user_data=self.y_path_id)
                with dpg.group(horizontal=True):  # Label on the left
                    dpg.add_text("Run From previous configurations: ")
                    self.run_path_id = dpg.add_input_text(tag="run_path",width=320)
                    dpg.add_button(label="browse", callback=self.load_callback_run, user_data=self.run_path_id)
                    dpg.add_button(label="RUN HISTORY", callback=self.run_from_history)
                dpg.add_button(label="Load", callback=self.load_files)
    
    def create_popup(self, text, window):
        dpg.hide_item(window)
        with dpg.window( tag="pop_up",show=True, width=400, height=200):
            with dpg.group(horizontal=False):
                dpg.add_text(text, tag="popup_text")
                dpg.add_button(label="OK", width=75,tag="popup_button", callback=lambda: self.popup_button(window))
            
        
        
                
                

    def setup_attack_defense_window(self):
        if dpg.does_item_exist("Select Attacks and Defenses"):
            dpg.delete_item("Select Attacks and Defenses")
        with dpg.window(label="Select Attacks and Defenses",tag="Select Attacks and Defenses", width=1280, height=720, no_move=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Select Attacks")
                for attack in supported_attacks:
                    supported_lib = supported_attacks[attack]["applicable_to"]
                    if dpg.get_value(self.library_id) in supported_lib:
                        dpg.add_checkbox(label=attack, tag=f"attack_{attack}")

                dpg.add_button(label="Select All Attacks", callback=lambda: self.select_all(True, supported_attacks, prefix="attack_"))
                dpg.add_button(label="Select None Attacks", callback=lambda: self.select_all(False, supported_attacks, prefix="attack_"))

                dpg.add_separator()

                dpg.add_text("Select Defenses")
                for defense in supported_defenses:
                    supported_lib = supported_defenses[defense]["applicable_to"]
                    if dpg.get_value(self.library_id) in supported_lib:
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
                                                                    "Configure Manually Attack and Defense Parameters"],tag="parameter_config", 
                                                                    horizontal=False)
                dpg.add_button(label="Proceed",tag="proceed_with_selection", callback=self.on_proceed_with_selection, user_data="Select Attacks and Defenses")
                dpg.add_button(label="back", callback=self.back_to_loading)
        
        



    def setup_parameter_configuration_window(self, selected_attacks, selected_defenses):
        if dpg.does_item_exist("Configure Parameters"):
            dpg.delete_item("Configure Parameters")
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
                                        print(f"{attack_config['name']}_{param}")
                                else:
                                    dpg.add_input_text(default_value=str(value), width=200, tag=f"{attack_config['name']}_{param}")
                                dpg.add_separator()

            dpg.add_separator()

            # Configure Defense Parameters
            with dpg.group(horizontal=False):
                dpg.add_text("Configure Defense Parameters")
                for defense_config in selected_defenses:
                    dpg.add_text(defense_config["name"])  # Display the name of the defense
                    for param, value in defense_config.items():
                        # Exclude non-parameter items
                        if param not in ["name", "applicable_to","attack_type"]:
                            with dpg.group(horizontal=True):  # Grouping label and input horizontally
                                dpg.add_text(f"{param}: ")  # Fixed width for the label
                                # Shorter input field with specific width
                                if isinstance(value,bool):
                                    dpg.add_checkbox(default_value=value, tag=f"{defense_config['name']}_{param}")
                                else:
                                    dpg.add_input_text(default_value=str(value), width=200, tag=f"{defense_config['name']}_{param}")
                                dpg.add_separator()

            dpg.add_separator()
            dpg.add_button(label="Begin Run", callback=self.on_proceed_with_selection_manual_config)
            dpg.add_button(label="back",callback=self.back_to_selection)
        

    def popup_button(self, window_tag):
        dpg.show_item(window_tag)
        dpg.delete_item("pop_up")
    def on_proceed_with_selection(self, sender, app_data, user_data):
        selected_option = dpg.get_value(self.param_config_option)
        if(selected_option == ""):# if nothing is selected by clicking string is empty instead of default run
            selected_option = "Run on Default Parameters"
        self.selected_attacks = [supported_attacks[attack] for attack in supported_attacks if dpg.get_value(f"attack_{attack}")]
        self.selected_defenses = [supported_defenses[defense] for defense in supported_defenses if dpg.get_value(f"defense_{defense}")]
        if len(self.selected_attacks)==0 and len(self.selected_defenses)==0:
            print(user_data)
            self.create_popup("Please select at least 1 attack or defence to apply",user_data)

        else:
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
                dpg.add_button(label="ANOTHER RUN",tag="reset", callback=self.reset_button, height=30, width= 60, show=False)
                dpg.add_spacer()  # Ensures the text widget is centered
                
            
            dpg.add_spacer(height=10)  # Adjust height for better vertical alignment
            

    def create_history(self):
        # Get the path to the user's home directory
        home_directory = os.path.dirname(os.path.abspath(__file__))
        home_directory = os.path.join(home_directory, "..")
        home_directory = os.path.join(home_directory, "..")
        print(home_directory)
        # Specify the folder and file name within the home directory
        folder_name = "run_history"
        current_timestamp = datetime.now()
        formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M")
        filename = f"history_{formatted_timestamp}.txt"
        folder_path = os.path.join(home_directory, folder_name)
        print(folder_path)
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        # Ensure the directory exists
        os.makedirs(folder_path, exist_ok=True)
        model_path = dpg.get_value(self.model_path_id)
        x_path = dpg.get_value(self.x_path_id)
        y_path = dpg.get_value(self.y_path_id)
        model_library = dpg.get_value(self.library_id)
        with open(file_path, 'w') as file:
            file.write(f"model_path  :{model_path}\n")
            file.write(f"model_library  :{model_library}\n")
            file.write(f"x_path  :{x_path}\n")
            file.write(f"y_path  :{y_path}\n\n")
            for attack_config in self.selected_attacks:
                file.write("attack_"+str(attack_config["name"])+"\n")  # Display the name of the attack
                for param, value in attack_config.items():
                    # Exclude non-parameter items
                    if param not in ["name", "type", "applicable_to"]:
                        file.write(f"{attack_config['name']}_{param}  :")  # Fixed width for the label
                        if dpg.does_item_exist("Configure Parameters"):
                            value = dpg.get_value(f"{attack_config['name']}_{param}")
                        if isinstance(value,bool):
                                file.write(f"{value}\n")
                        else:
                            file.write(str(value)+"\n")
            for defence_config in self.selected_defenses:
                file.write("\ndefense_"+str(defence_config["name"])+"\n")  # Display the name of the attack
                for param, value in defence_config.items():
                    # Exclude non-parameter items
                    if param not in ["name", "attack_type", "applicable_to"]:
                        file.write(f"{defence_config['name']}_{param}  :")  # Fixed width for the label
                        if dpg.does_item_exist("Configure Parameters"):
                            value = dpg.get_value(f"{defence_config['name']}_{param}")
                        if isinstance(value,bool):
                                file.write(f"{value}\n")
                        else:
                            file.write(str(value)+"\n")
                        file.write

    def run_from_history(self):
        self.from_history = True
        clean_lines = []
        found = False
        try:
            with open(dpg.get_value(self.run_path_id), "r") as file:
        # Iterate through each line in the file
                for line in file:
                    if len(clean_lines) == 4 and found == False:
                        dpg.set_value(self.model_path_id,clean_lines[0][1]) 
                        dpg.set_value(self.library_id,clean_lines[1][1])
                        dpg.set_value(self.x_path_id,clean_lines[2][1])
                        dpg.set_value(self.y_path_id,clean_lines[3][1])
                        self.load_files()
                        clean_lines = []
                        found = True
                    line = line.strip()
                    parts = line.split(":")
                    if parts[0] == "":
                        continue
                    if len(parts) > 1:
                        parts[0] = parts[0].replace(" ", "")
                        clean_lines.append(parts)
                    if len(parts) == 1:
                        try:
                            dpg.set_value(parts[0], True)
                        except Exception as e:
                            self.create_popup("Please Choose a valid configurations file","Main_Window" )
                            dpg.delete_item("Select Attacks and Defenses")
                            self.run_from_history = False
                            return
        except Exception as e:
            self.create_popup("Please Choose a valid configurations file","Main_Window" )
            self.run_from_history = False
            return
        dpg.set_value(self.param_config_option, "Configure Manually Attack and Defense Parameters")
        self.on_proceed_with_selection(None,None,"Select Attacks and Defenses")
        for line in clean_lines:
            try:
                val = dpg.get_value(line[0])
                if isinstance(val,bool):
                    if line[1] == "True":
                        dpg.set_value(line[0],True)
                    else:
                        dpg.set_value(line[0],False)
                else:
                    dpg.set_value(line[0],line[1])
            except Exception as e:
                self.back_to_selection()
                self.back_to_loading()
                dpg.delete_item("Select Attacks and Defenses")
                self.create_popup("Please Choose a valid configurations file","Main_Window" )
                self.run_from_history = False
                return
        self.on_proceed_with_selection_manual_config()
        
    def update_progress(self, message, done = False):
        if dpg.does_item_exist("progress_text"):
            dpg.set_value("progress_text",dpg.get_value("progress_text")+'\n'+ message)
            if done:
                if not self.from_history:
                    self.create_history()
                dpg.show_item("reset")

        else:
            print("Progress window is not open or progress text does not exist.")


    def select_all(self, select, items, prefix):
        for item in items:
            dpg.set_value(f"{prefix}{item}", select)
    
    def back_to_selection(self):
        dpg.hide_item("Configure Parameters")
        dpg.show_item("Select Attacks and Defenses")

    def load_files(self):
        model_path = dpg.get_value(self.model_path_id)
        x_path = dpg.get_value(self.x_path_id)
        y_path = dpg.get_value(self.y_path_id)
        selected_library = supported_libraries[dpg.get_value(self.library_id)]

        if self.controller:
            success = None
            try:
                success = self.controller.handle_load(model_path, x_path, y_path, selected_library)
                if success:
                    dpg.hide_item("Main_Window")  # Assuming "Main Window" is the tag of the main window
                    if not dpg.does_item_exist("Select Attacks and Defenses"):
                        self.setup_attack_defense_window()
                    else:
                        dpg.show_item("Select Attacks and Defenses")
                # else:
                #     self.create_popup("Loading unsuccessful, please choose appropriate files","Main_Window")
            except Exception as e:
                if self.from_history:
                    self.create_popup("Please choose a valid configurations file","Main_Window")
                else:
                    self.create_popup("Loading unsuccessful, please choose appropriate filessss","Main_Window")
        else:
            print("Controller not set!")
    
    def back_to_loading(self):
        if self.controller:
                dpg.hide_item("Select Attacks and Defenses")  # Assuming "Main Window" is the tag of the main window
                dpg.show_item("Main_Window")
        else:
            print("Controller not set!")
    
    def reset_button(self):
        dpg.delete_item("progress_window")
        dpg.delete_item("Main_Window")
        dpg.delete_item("Select Attacks and Defenses")
        dpg.delete_item("Configure Parameters")
        self.setup_ui()
    
    def run(self):
        dpg.create_viewport(title='Custom Title', width=1280, height=720)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
