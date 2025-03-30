import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor


def load_and_preprocess(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    processed_data = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for _, row in df.iterrows():
        text = row['text']
        image_path = row['image_path'] if 'image_path' in row else 'N/A'
        label = row['label']  # Assume label is already 0 or 1

        img = None
        if image_path and os.path.exists(os.path.join(image_dir, image_path)):
            img = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
            img = transform(img)

        processed_data.append({
            'text': text,
            'image': img,
            'label': label
        })

    return processed_data

