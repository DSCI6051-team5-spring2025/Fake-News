import torch
from PIL import Image
import os

def predict(model, processor, text=None, image_path=None, device='cpu'):
    if not text and not image_path:
        raise ValueError("Either text or image path must be provided.")

    if image_path and os.path.exists(image_path) and image_path != 'N/A':
        image = Image.open(image_path).convert("RGB")
    elif image_path:
        print(f"Image not found: {image_path}. Using placeholder.")
        image = Image.new('RGB', (224, 224), color='white')
    else:
        image = None

    with torch.no_grad():
        if text and image:
            inputs = processor(text=text, images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(device)
            logits, _ = model(**inputs, return_loss=False)

        elif text:
            inputs = processor(text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(device)
            text_features = model.clip.get_text_features(**inputs)
            logits = model.classifier(text_features)

        elif image:
            inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = model.clip.get_image_features(**inputs)
            logits = model.classifier(image_features)

        else:
            raise ValueError("Both text and image are missing!")

    probs = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label].item()

    label_str = "Real" if pred_label == 1 else "Fake"
    return label_str, confidence

