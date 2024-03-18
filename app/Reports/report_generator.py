import app.Reports.json_to_pdf_converter as jtp
import json

class Report_Generator:
    def __init__(self, all_metrics):
            self.all_metrics = all_metrics
            self.file = "metrics.json"

    def dict_preperation(self):
        to_be_changed = []
        for key in self.all_metrics.keys():
            if type(key) != str:
                to_be_changed.append(key)
        for key in to_be_changed:
            self.all_metrics[str(key)] = self.all_metrics[key]
            del self.all_metrics[key]
            
    def build_json(self):
        self.dict_preperation()
        #replacing all the values with the dictionary of the metrics
        # for key in self.all_metrics.keys():
        #     self.all_metrics[key] = self.all_metrics[key].get_metrics()
             
        # Writing dictionary to JSON file
        with open(self.file, "w") as json_file:
            json.dump(self.all_metrics, json_file)

    def generate_pdf(self, data, adv_examples):
         self.build_json()
         pdf = jtp.Json_To_Pdf(self.file, data, adv_examples)
         return pdf.create_pdf()