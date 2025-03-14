import torch
import requests
import pandas as pd
from PIL import Image
import open_clip
from io import BytesIO
import google.generativeai as genai

# 🔹 Replace with your Google AI API key
genai.configure(api_key="AIzaSyDHUT0g_9s_oIuqk6H5lvP-hzR6yPJ683s")

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = open_clip.create_model("ViT-B-32", pretrained="openai").to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
preprocess = open_clip.image_transform(model.visual.image_size, is_train=False)

# Load the dataset
df = pd.read_csv("politifact_clip_labeled_dataset.csv")

# Function to download and preprocess image
def process_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return preprocess(image).unsqueeze(0).to(device)
        else:
            print(f" Skipping (Status Code {response.status_code}): {image_url}")
            return None
    except Exception as e:
        print(f" Error fetching image: {image_url} - {e}")
        return None

# Function to classify image-text relationship using CLIP
def classify_with_clip(claim, image_url):
    image_tensor = process_image(image_url)
    if image_tensor is None:
        return "Error", 0

    # Tokenize and encode text
    text_tensor = tokenizer([claim]).to(device)

    # Get CLIP embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)

        # Normalize embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity score
        similarity = (image_features @ text_features.T).item()

    # Set a threshold for classification
    label = "Related" if similarity > 0.25 else "Unrelated"
    return label, similarity

# Function to generate explanation using Gemini
def generate_explanation(claim, image_url, similarity_score, label):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Using Gemini API
        response = model.generate_content([
            {"text": f"Analyze the relationship between the claim: '{claim}' and the given image (URL: {image_url}). "
                     f"The AI model (CLIP) has classified this image as '{label}' with a similarity score of {similarity_score:.2f}. "
                     "Explain in detail why this classification was made, considering any visible elements, objects, and text in the image."}
        ])

        return response.text if response.text else "No explanation provided"
    except Exception as e:
        print(f" Error generating explanation for {image_url}: {e}")
        return "Error generating explanation"

# Apply CLIP model and generate explanations
df["CLIP_Label"], df["CLIP_Similarity"] = zip(*df.apply(lambda row: classify_with_clip(row["Claim"], row["Image URL"]), axis=1))

# Generate explanations for each image
df["Explanation"] = df.apply(lambda row: generate_explanation(row["Claim"], row["Image URL"], row["CLIP_Similarity"], row["CLIP_Label"])
                             if row["CLIP_Label"] != "Error" else "No explanation (Error)", axis=1)

# Save the updated dataset with explanations
df.to_csv("politifact_clip_explanations.csv", index=False)

print(" CLIP Labels with Explanations Generated & Saved!")
