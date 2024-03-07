from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import json

def is_numeric_array(arr):
    return all(isinstance(item, (int, float)) for item in arr)

def add_json_to_pdf(content, json_data, indent=0):
    styles = getSampleStyleSheet()
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, dict):
                content.append(Paragraph("<b>{}</b>: ".format(key), styles["Normal"]))
                add_json_to_pdf(content, value, indent + 1)
            elif isinstance(value, list):
                if is_numeric_array(value):
                    content.append(Paragraph("<b>{}</b>: {}".format(key, value), styles["Normal"]))
                else:
                    content.append(Paragraph("<b>{}</b>:".format(key), styles["Normal"]))
                    for item in value:
                        if isinstance(item, dict):
                            add_json_to_pdf(content, item, indent + 1)
                        else:
                            content.append(Paragraph(str(item), styles["Normal"]))
            else:
                content.append(Paragraph("<b>{}</b>: {}".format(key, value), styles["Normal"]))
    elif isinstance(json_data, list):
        if is_numeric_array(json_data):
            content.append(Paragraph(str(json_data), styles["Normal"]))
        else:
            for item in json_data:
                if isinstance(item, dict):
                    add_json_to_pdf(content, item, indent)
                else:
                    content.append(Paragraph(str(item), styles["Normal"]))

def create_pdf(json_path, output_pdf):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    title = Paragraph("<b>Results:</b>", styles["Title"])
    content.append(title)
    content.append(Spacer(1, 12))

    add_json_to_pdf(content, json_data)
    doc.build(content)

# Example usage:
json_path = "data.json"
output_pdf = "output.pdf"
create_pdf(json_path, output_pdf)

