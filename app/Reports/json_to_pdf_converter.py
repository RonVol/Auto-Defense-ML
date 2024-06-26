from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import numpy as np
from matplotlib import pyplot as plt
import os
import json
import datetime
import uuid

current_time = datetime.datetime.now()
timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
unique_id = str(uuid.uuid4()).split('-')[0]
filename = f"{timestamp_str}_{unique_id}"

class Json_To_Pdf:
    def __init__(self, json_file, data, adv_examples):
        self.json_file = json_file
        self.output_pdf = os.getcwd()+"/Reports/"+filename+self.json_file[:-5]+".pdf"
        self.data = data
        self.adv = adv_examples

    def is_numeric_array(arr):
        return all(isinstance(item, (int, float)) for item in arr)

    def get_data(self, num_classes=200):
        x_train = self.data.x
        y_train = self.data.y
        #y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1],)[:x_train.shape[0]]
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
        #print(self.data.model)
        fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))
        colors = ['blue', 'green', 'red', 'purple']
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
            Z_proba = self.data.model.predict_proba(np.c_[xx.all(), yy.all(), xx.any(), yy.any()])
            #print(Z_proba, Z_proba.shape)
            Z_proba = Z_proba[:, i_class]
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

    def add_json_to_table(self, json_data):
        data = []
        self.parse_json_to_table(json_data, data)
        return data

    def parse_json_to_table(self, json_data, data, prefix=""):
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    data.append([Paragraph("<b>{}</b>".format(key), self.styles["BodyText"]), ''])
                    self.parse_json_to_table(value, data)
                else:
                    data.append([prefix + key, str(value)])
        elif isinstance(json_data, list):
            for i, item in enumerate(json_data):
                self.parse_json_to_table(item, data, prefix= f"{i}.")

    def graphs(self, data, content, filename):
        # Extract necessary data
        methods = list(data.keys())
        accuracy_values = [data[key]['overall_accuracy'] for key in methods]
        precision_values = [data[key]['overall_precision'] for key in methods]
        recall_values = [data[key]['overall_recall'] for key in methods]

        # Create bar graph
        plt.figure(figsize=(10, 5))

        plt.bar(methods, accuracy_values, color='b', width=0.2, align='center', label='Accuracy')
        plt.bar(methods, precision_values, color='g', width=0.2, align='edge', label='Precision')
        plt.bar(methods, recall_values, color='r', width=0.2, align='edge', label='Recall')

        #plt.xlabel('Methods')
        plt.ylabel('Scores')
        plt.title('Comparison of Methods')

        plt.legend()
        plt.grid(False)
        plt.xticks(rotation=75)

        plt.tight_layout()

        # Save the graph as an image
        plt.savefig(filename + '.png')
        content.append(Image(filename + '.png', width=600))
        return content

    def process_phrase(self, phrase):
        # If the phrase is enclosed in parentheses, remove them
        if phrase.startswith("(") and phrase.endswith(")"):
            phrase = phrase[1:-1]
        return phrase

    def group_similar_phrases(self, phrases):
        phrase_groups = defaultdict(list)
        
        for phrase in phrases:
            phrase = self.process_phrase(phrase)
            words = sorted(phrase.split())
            key = tuple(words)
            # Check if the phrase contains commas
            if ',' in phrase:
                phrase_groups[key].append(phrase)
            else:
                phrase_groups[key].append([phrase])
        
        return list(phrase_groups.values())



    def create_pdf(self):
        #print("output path:", self.output_pdf)
        with open(self.json_file, 'r') as f:
            json_data = json.load(f)

        doc = SimpleDocTemplate(self.output_pdf, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.styles = getSampleStyleSheet()
        content = []

        title = Paragraph("<b>Evaluation Report:</b>", self.styles["Title"])
        content.append(title)
        content.append(Spacer(1, 12))

         # Convert JSON data to table format
        table_data = self.add_json_to_table(json_data)

        # Create a table with two columns
        table = Table(table_data, colWidths=[300, 300])

        # Define table style
        style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)])

        # Apply table style
        table.setStyle(style)
        content.append(table)
        content = self.graphs(json_data, content, 'All')
        methods = list(json_data.keys())
        attacks = self.group_similar_phrases(methods)
        print("methods", methods)
        print("attacks", attacks)
        for attack in attacks:
            for options in attack:
                for option in options:
                    print("attack:" , option)
                    compare = {}
                    if option == 'Clean':
                        continue
                    elif option in methods:
                        compare[option] = json_data[option]
                #content = self.graphs(compare, content, option)
        doc.build(content)

    # # Example usage:
    # json_file = "data.json"
    # output_pdf = "output.pdf"
    # create_pdf(json_file, output_pdf)

