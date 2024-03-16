import utils.json_to_pdf_converter as jtp
import json

class Report_Generator:
    def __init__(self, all_metrics):
            self.all_metrics = all_metrics
            self.pdf = "metrics.json"

    def build_json(self):
        #replacing all the values with the dictionary of the metrics
        for key in self.all_metrics.keys:
             self.all_metrics[key] = self.all_metrics[key].run_metrics_calculations()

        # Writing dictionary to JSON file
        with open(self.file, "w") as json_file:
            json.dump(self.all_metrics, json_file)

    def generate_pdf(self):
         jtp(self.file)
         return jtp.create_pdf()