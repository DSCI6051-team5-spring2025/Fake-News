from fastapi import FastAPI, UploadFile, Form
from src.predict import predict
from transformers import CLIPProcessor
from src.model import CLIPClassifier
import torch
from PIL import Image
import io

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPClassifier()
model.load_state_dict(torch.load("./models/clip_fake_news_classifier1.pth", map_location=device))
model.to(device)
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.post("/predict/")
async def get_prediction(text: str = Form(None), image: UploadFile = None):
    image_path = None
    if image:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_path = "./temp_image.jpg"
        img.save(image_path)

    label, confidence = predict(model, processor, text=text, image_path=image_path, device=device)
    return {"label": label, "confidence": confidence}

