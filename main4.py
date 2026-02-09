import torch
import requests
from PIL import Image
from torchvision import transforms
import gradio as gr

# 1. Use 'main' to avoid the v0.6.0 error; use 'weights' for modern compatibility
model = torch.hub.load('pytorch/vision:main', 'resnet18', weights='ResNet18_Weights.DEFAULT').eval()

# 2. Proper ImageNet transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download labels
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp):
    # Apply the preprocessing pipeline
    inp = preprocess(inp).unsqueeze(0) 
    
    with torch.no_grad():
        # Get predictions
        output = model(inp)
        prediction = torch.nn.functional.softmax(output[0], dim=0)
        
        # Create dictionary of labels and probabilities
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences

# 4. Gradio Interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    # Ensure these files actually exist in your path or remove the examples line
    examples=["lion.jpg", "cheetah.jpg"] 
).launch()