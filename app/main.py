from UI.main_ui import run_ui
from Core import helper
from Core.controller import Controller


# Now you can use imports relative to the script_dir
from Core import controller

if __name__ == "__main__":
    # run_ui()
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = helper.get_mnist()
    classifier = helper.get_mnist_cnn_classifier()
    controller = Controller(classifier=classifier,x_test=x_test,y_test=y_test)
    controller.run()
    
