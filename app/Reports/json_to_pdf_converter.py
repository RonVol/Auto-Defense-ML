from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import numpy as np
import re
from matplotlib import pyplot as plt
from collections import Counter
import os
import json
import datetime
import uuid
import platform

current_time = datetime.datetime.now()
timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M")
unique_id = str(uuid.uuid4()).split('-')[0]
filename = "metrics"+timestamp_str+"-"+unique_id

def get_os_type():
    os_type = platform.system()
    if os_type == "Windows":
        return "Windows"
    elif os_type == "Darwin":
        return "macOS"
    elif os_type == "Linux":
        return "Linux"
    else:
        return "Unknown"
    
class Json_To_Pdf:
    def __init__(self, json_file, data, adv_examples):
        self.json_file = json_file
        self.output_pdf = os.path.dirname(os.path.realpath(filename))+"/app/Reports/"+filename+".pdf"
        self.data = data
        self.adv = adv_examples
        self.path = os.path.dirname(os.path.abspath(__file__))+"/"+filename+".pdf"
        with open(self.json_file, 'r') as f:
            json_data = json.load(f)
        methods = list(json_data.keys())
        self.attacks = [s for s in methods if "Attack" in s and "(" not in s]
        if "HopSkipJump" in methods:
            self.attacks.append("HopSkipJump")
        self.defenses = [x for x in methods if x not in self.attacks and "(" not in x]
        self.defenses.remove("Clean")
        
    def open_pdf(self):
            #try:
            system = get_os_type()
            if os.path.exists(self.path):
                # Open the PDF file with the default system viewer
                if system == "Windows":
                    os.startfile(self.path)  # For Windows
                if system == "macOS":
                    os.system(f'open "{self.path}"')
                if system == "Linux":
                    os.system(f'xdg-open "{self.path}"')
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
        # #print("model: ", model)
        # #print("x_train: ", x_train)
        # #print("y_train: ", y_train)
        # #print("x_train_adv: ", x_train_adv)
        # #print("num_classes: ", num_classes)
        x_train_adv = x_train_adv[list(x_train_adv)[0]]
        ##print(self.data.model)
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
            ##print(Z_proba, Z_proba.shape)
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

    def graphs(self, data, content, filename=str(uuid.uuid4()).split('-')[0]):
        # Extract necessary data
        methods = list(data.keys())
        accuracy_values = [data[key]['overall_accuracy'] for key in methods]
        precision_values = [data[key]['overall_precision'] for key in methods]
        recall_values = [data[key]['overall_recall'] for key in methods]

        # Create bar graph
        plt.figure(figsize=(10, 5))

        plt.bar(methods, accuracy_values, color='b', width=0.15, align='center', label='Accuracy')
        plt.bar(np.arange(len(methods))+0.15, precision_values, color='g', width=0.15, align='edge', label='Precision')
        plt.bar(methods, recall_values, color='r', width=0.15, align='edge', label='Recall')

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

    def remove_duplicates(self, list_of_lists):
        seen = set()
        result = []
        
        for lst in list_of_lists:
            # Convert list to tuple since lists are unhashable (cannot be used as set keys)
            tpl = tuple(lst)
            if tpl not in seen:
                seen.add(tpl)
                result.append(lst)
        
        return result
    
    def recommend(self, data, json):
        #print("data:", data)
        methods = data
        ##print("compare:", json)
        results = {}
        #print("methods:", methods)
        #methods = list(data.keys())
        accuracy_values = [(key, json[key]['overall_accuracy']) for key in methods]
        precision_values = [(key, json[key]['overall_precision']) for key in methods]
        recall_values = [(key, json[key]['overall_recall']) for key in methods]
        #print("acc:", accuracy_values)
        #print("pres:", precision_values)
        #print("rec:", recall_values)
        accuracy_values.sort(key = lambda x: x[1], reverse=True)
        precision_values.sort(key = lambda x: x[1], reverse=True)
        recall_values.sort(key = lambda x: x[1], reverse=True)
        results["acc"] = [x for x in accuracy_values]
        results["pres"] = [x for x in precision_values]
        results["rec"] = [x for x in recall_values]
        return results

    def create_pdf(self):
        ##print("output path:", self.output_pdf)
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
        if self.defenses == []:
            title = Paragraph("<b>Recommendations:</b>", self.styles["Title"])
            content.append(title)
            content.append(Spacer(1, 12))
            content.append(Paragraph("There were no defenses selected, therefore there is no recommendation\n", self.styles["BodyText"]))
        else:
            methods = list(json_data.keys())
            results = {}
            grouped = []
            for method in methods:
                for other_method in methods:
                    if method == other_method:
                        continue
                    else:
                        if other_method in method:
                            grouped.append([method, other_method, 'Clean'])
            #print("\n")
            #print("groupped:", grouped)
            grouped_lists = defaultdict(list)
            for inner_list in grouped:
                key = inner_list[0]  # Use the first element as the key for grouping
                grouped_lists[key].append(inner_list)

            grouped_lists = dict(grouped_lists)
            for key, group in grouped_lists.items():
                grouped_lists[key] = set([item for sublist in group for item in sublist])
            #print("grouped_lists:", grouped_lists)
            grouped = list(grouped_lists.values())
            #print("grouped after:", grouped)
            for vs in grouped:
                compare = {}
                for run in vs:
                    compare[run] = json_data[run]
                ##print("compare:", compare.keys())
                content = self.graphs(compare, content, run+str(uuid.uuid4()).split('-')[0])
                key = []
                res = {}
                for x in grouped:
                    for y in x:
                        if y in self.attacks:
                            if y in key:
                                res[y] = res[y] + [x]
                            else:
                                res[y] = [x]
                            key.append(y)
                merged_dict = {}
                for key, value_list in res.items():
                    merged_set = set()
                    for s in value_list:
                        merged_set.update(s)
                    
                    merged_dict[key] = list(merged_set)
            #print("merged:", merged_dict)
            title = Paragraph("<b>Recommendations:</b>", self.styles["Title"])
            content.append(title)
            content.append(Spacer(1, 12))
            content.append(Paragraph("According to the Results of the automation:\n", self.styles["BodyText"]))
        
            for key in merged_dict.keys():
                #print("\n")
                #print("key:", key)
                #print("vals:", merged_dict.values())
                #print("compare:", json_data.keys())
                for run in merged_dict.values():
                    if key in run:
                        results[key] = self.recommend(run, json_data)
            #print("\n")
            #print("results:", results)
            suggested = {
                outer_key: {
                    inner_key: [tup[0] for tup in inner_value if tup[0] != "Clean" and tup[0] not in self.defenses]
                    for inner_key, inner_value in inner_dict.items()
                }
                for outer_key, inner_dict in results.items()
            }
            recc = {}
            for outer_key, inner_dict in suggested.items():
                suggestion = []
                better_than_the_rest = False
                for inner_key, inner_value in inner_dict.items():
                    #print("inner value:", inner_value)
                    for method in inner_value:
                        #print("method:", method)
                        if method == "Clean" or better_than_the_rest == True:
                            continue
                        else:
                            if method in self.attacks:
                                better_than_the_rest = True
                                recc[method] = suggestion
                            else:
                                suggestion.append(method)
                #print("\n")
                #print("suggestion:", suggestion)
                #print("recc:", recc)
                for method, val in recc.items():
                    if len(val) == 1:
                        val = val[0].strip("(\')").split(",")
                        #print(val)
                        content.append(Paragraph(f"The reccomended action is to use {val[0][:-2]} against {val[1][2:]}\n", self.styles["BodyText"]))
                    else:
                        acc = [x[0] for x in results[method]["acc"] if x[0] not in self.attacks and x[0] != "Clean"]
                        pres = [x[0] for x in results[method]["pres"] if x[0] not in self.attacks and x[0] != "Clean"]
                        rec = [x[0] for x in results[method]["rec"] if x[0] not in self.attacks and x[0] != "Clean"]
                        #print(acc, pres, rec)
                        # index_counts = defaultdict(lambda: defaultdict(int))
        
                        # # Iterate over each list
                        # for idx, lst in enumerate([acc, pres, rec]):
                        #     # Iterate over each element in the list
                        #     for i, elem in enumerate(lst):
                        #         index_counts[i][elem] += 1  
                        # for index, counts in index_counts.items():
                        #     #print(f"At index {index}:")
                        #     for elem, count in counts.keys():
                        #         #print(f"Element {elem}: {count} times")
                        defence = [x for x in results[method]["acc"] if x[0] in self.defenses]
                        #print(defence)
                        comb = [x for x in results[method]["acc"] if x[0] not in self.defenses and x[0] not in self.attacks and x[0] != "Clean"]
                            
                        #print(comb[0], comb[1])#[results[method]["acc"].index(val1)][1])#,results[method]["acc"][results[method]["acc"].index(val2)][1])
                        if comb[0][1] > comb[1][1]:
                            val1 = comb[0][0].strip("(\')").split(",")[0][:-1]
                            val2 = comb[1][0].strip("(\')").split(",")[0][:-1]
                            ##print(val1, val2)
                            content.append(Paragraph(f"The first reccomended action is to use{val1} against {method}\n", self.styles["BodyText"]))
                            content.append(Paragraph(f"The second reccomended action is to use {val2} against {method}\n", self.styles["BodyText"]))
        doc.build(content)
        for filename in os.listdir(os.getcwd()):
            if filename.endswith('.png'):
                os.remove(os.path.join(os.getcwd(), filename))
        self.open_pdf()


    # # Example usage:
    # json_file = "data.json"
    # output_pdf = "output.pdf"
    # create_pdf(json_file, output_pdf)

