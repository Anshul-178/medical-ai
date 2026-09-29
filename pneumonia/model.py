import torch
import torch.nn as nn
from torchvision import models,transforms
from torchvision.models import resnet18
from PIL import Image
from pathlib import Path


model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 2)
checkpoint=torch.load(Path(__file__).parent / 'best_model.pth',map_location='cpu')
#print(checkpoint)
model.load_state_dict(checkpoint["model_state_dict"]) 
model.eval()


transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

classes=["Normal","Pneumonia"]

# Image path
def predict_pneumonia(image):
    image=image.convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output,1).item()

    return classes[pred]
