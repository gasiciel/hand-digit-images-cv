import gradio as gr
import torch
import torchvision
from PIL import Image
import numpy as np

class_names = ['anchor', 'balloon', 'bicycle', 'envelope', 'paper_boat', 'peace_symbol', 'smiley', 'speech_bubble', 'spiral', 'thumb']

model = torchvision.models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))

model.load_state_dict(torch.load('resnet18.pth'))
model.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(input_data):
    if input_data is None:
        return {}

    if isinstance(input_data, dict):
        input_image = Image.fromarray(input_data['composite'])
    elif isinstance(input_data, np.ndarray):
        input_image = Image.fromarray(input_data)
    else:
        input_image = input_data

    input_image = input_image.convert("RGB")
        
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    
    probs = torch.nn.functional.softmax(output[0], dim=0)
    conf = {class_names[i]: float(prob) for i, prob in enumerate(probs)}
    
    return conf


with gr.Blocks() as interface:
    gr.Markdown("# Klasyfikator ręcznie rysowanych i zdigitalizowanych obrazków")
    gr.Markdown(
        """
        Wgraj obraz lub przejdź do zakładki 'Narysuj obrazek', aby narysować coś samemu!
        Model został wytrenowany na zbiorze danych [Simple hand-drawn and digitized images](https://www.kaggle.com/datasets/gergvincze/simple-hand-drawn-and-digitized-images/data)
        """
    )

    with gr.Tabs():
        with gr.TabItem("Wgraj obraz"):
            image_input = gr.Image(type="pil", label="Wgraj plik")
            upload_button = gr.Button("Klasyfikuj")
            
        with gr.TabItem("Narysuj obrazek"):
            gr.Markdown("**Uwaga:** Funkcja rysowania jest wersją demonstracyjną i prawdopodobnie nie działa zbyt dobrze:)")
            sketch_input = gr.Paint(type="numpy", label="Pole do rysowania")
            draw_button = gr.Button("Klasyfikuj")

    output_label = gr.Label(num_top_classes=3, label="Najbardziej prawdopodobne klasy")

    upload_button.click(fn=predict, inputs=image_input, outputs=output_label)
    draw_button.click(fn=predict, inputs=sketch_input, outputs=output_label)

interface.launch()