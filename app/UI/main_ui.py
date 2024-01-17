import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

def run_ui():
    dpg.create_context()
    dpg.create_viewport(title='A Title', width=600, height=600)

    demo.show_demo()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()