from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
from matplotlib import pyplot as plt
import os
import json

class Json_To_Pdf:
    def __init__(self, json_file, data, adv_examples):
        self.json_file = json_file
        self.output_pdf = os.getcwd()+"/"+self.json_file[:-5]+".pdf"
        self.data = data
        self.adv = adv_examples

    def is_numeric_array(arr):
        return all(isinstance(item, (int, float)) for item in arr)

    def get_data(self, num_classes=200):
        x_train = self.data.x
        y_train = self.data.y
        y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1],)[:x_train.shape[0]]
        x_train = x_train[y_train < num_classes][:, [0, 1]]
        y_train = y_train[y_train < num_classes]
        x_train[:, 0][y_train == 0] *= 2
        x_train[:, 1][y_train == 2] *= 2
        x_train[:, 0][y_train == 0] -= 3
        x_train[:, 1][y_train == 2] -= 2
        
        x_train[:, 0] = (x_train[:, 0] - 4) / (9 - 4)
        x_train[:, 1] = (x_train[:, 1] - 1) / (6 - 1)
        
        return x_train, y_train

    def plot_results(self, x_train, y_train, x_train_adv, num_classes=2):
        # print("model: ", model)
        # print("x_train: ", x_train)
        # print("y_train: ", y_train)
        # print("x_train_adv: ", x_train_adv)
        # print("num_classes: ", num_classes)
        x_train_adv = x_train_adv[list(x_train_adv)[0]]
        print(self.data.model)
        fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))
        colors = ['orange', 'blue', 'green']
        for i_class in range(num_classes):
            # Plot difference vectors
            for i in range(y_train[y_train == i_class].shape[0]):
                x_1_0 = x_train[y_train == i_class][i, 0]
                x_1_1 = x_train[y_train == i_class][i, 1]
                x_2_0 = x_train_adv[y_train == i_class][i, 0]
                x_2_1 = x_train_adv[y_train == i_class][i, 1]
                if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                    axs[i_class].plot([x_1_0, x_2_0], [x_1_1, x_2_1], c='black', zorder=1)

            # Plot benign samples
            for i_class_2 in range(num_classes):
                axs[i_class].scatter(x_train[y_train == i_class_2][:, 0], x_train[y_train == i_class_2][:, 1], s=20,
                                    zorder=2, c=colors[i_class_2])
            axs[i_class].set_aspect('equal', adjustable='box')

            # Show predicted probability as contour plot
            h = .01
            x_min, x_max = 0, 1
            y_min, y_max = 0, 1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            Z_proba = self.data.model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z_proba = Z_proba[:, i_class].reshape(xx.shape)
            im = axs[i_class].contourf(xx, yy, Z_proba, levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                    vmin=0, vmax=1)
            if i_class == num_classes - 1:
                cax = fig.add_axes([0.95, 0.2, 0.025, 0.6])
                plt.colorbar(im, ax=axs[i_class], cax=cax)

            # Plot adversarial samples
            for i in range(y_train[y_train == i_class].shape[0]):
                x_1_0 = x_train[y_train == i_class][i, 0]
                x_1_1 = x_train[y_train == i_class][i, 1]
                x_2_0 = x_train_adv[y_train == i_class][i, 0]
                x_2_1 = x_train_adv[y_train == i_class][i, 1]
                if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                    axs[i_class].scatter(x_2_0, x_2_1, zorder=2, c='red', marker='X')
            axs[i_class].set_xlim((x_min, x_max))
            axs[i_class].set_ylim((y_min, y_max))

            axs[i_class].set_title('class ' + str(i_class))
            axs[i_class].set_xlabel('feature 1')
            axs[i_class].set_ylabel('feature 2')

    def add_json_to_pdf(self, content, json_data, indent=0):
        styles = getSampleStyleSheet()
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                if isinstance(value, dict):
                    content.append(Paragraph("<b>{}</b>: ".format(key), styles["Heading2"]))
                    x_train, y_train = self.get_data()
                    self.plot_results(x_train, y_train, self.adv)
                    self.add_json_to_pdf(content, value, indent + 1)
                elif isinstance(value, list):
                    if self.is_numeric_array(value):
                        content.append(Paragraph("<b>{}</b>: {}".format(key, value), styles["Normal"]))
                    else:
                        content.append(Paragraph("<b>{}</b>:".format(key), styles["Normal"]))
                        for item in value:
                            if isinstance(item, dict):
                                self.add_json_to_pdf(content, item, indent + 1)
                            else:
                                content.append(Paragraph(str(item), styles["Normal"]))
                else:
                    content.append(Paragraph("<b>{}</b>: {}".format(key, value), styles["Normal"]))
        elif isinstance(json_data, list):
            if self.is_numeric_array(json_data):
                content.append(Paragraph(str(json_data), styles["Normal"]))
            else:
                for item in json_data:
                    if isinstance(item, dict):
                        self.add_json_to_pdf(content, item, indent)
                    else:
                        content.append(Paragraph(str(item), styles["Normal"]))

    def create_pdf(self):
        #print("output path:", self.output_pdf)
        with open(self.json_file, 'r') as f:
            json_data = json.load(f)

        doc = SimpleDocTemplate(self.output_pdf, pagesize=letter)
        styles = getSampleStyleSheet()
        content = []

        title = Paragraph("<b>Results:</b>", styles["Title"])
        content.append(title)
        content.append(Spacer(1, 12))

        self.add_json_to_pdf(content, json_data)
        doc.build(content)

    # # Example usage:
    # json_file = "data.json"
    # output_pdf = "output.pdf"
    # create_pdf(json_file, output_pdf)

