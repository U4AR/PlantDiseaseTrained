import io
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
from fastapi import FastAPI, File, UploadFile
from typing import Optional

# Define the device to use for computation
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the saved model
model = resnet50(pretrained=False)
num_classes = 38  # Update this based on the number of classes in your dataset
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load('plant_village_resnet50.pth', map_location=device))
model = model.to(device)
model.eval()

# Define class labels based on your dataset
class_to_idx = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3,
    'Blueberry___healthy': 4, 'Cherry_(including_sour)___healthy': 5, 'Cherry_(including_sour)___Powdery_mildew': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___healthy': 9, 'Corn_(maize)___Northern_Leaf_Blight': 10, 'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12, 'Grape___healthy': 13, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20,
    'Potato___healthy': 21, 'Potato___Late_blight': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25, 'Strawberry___healthy': 26, 'Strawberry___Leaf_scorch': 27,
    'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___healthy': 30, 'Tomato___Late_blight': 31,
    'Tomato___Leaf_Mold': 32, 'Tomato___Septoria_leaf_spot': 33, 'Tomato___Spider_mites Two-spotted_spider_mite': 34,
    'Tomato___Target_Spot': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 37
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Function to predict the class of an image
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)  # Apply transforms and add batch dimension
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = idx_to_class[predicted.item()]
    
    return predicted_class

# FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Predict the class of the image
    predicted_class = predict_image(image)
    
    return {"prediction": predicted_class}
