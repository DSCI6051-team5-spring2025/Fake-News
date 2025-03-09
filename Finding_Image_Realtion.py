import torch
import clip
from PIL import Image
import pandas as pd

# Load the model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the dataset
df = pd.read_csv("politifact_cleaned_dataset.csv")

# Function to calculate similarity between article text and image
def get_image_text_similarity(image_path, text):
    try:
        # Load and preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Tokenize text
        text = clip.tokenize([text]).to(device)

        # Encode image and text
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        # Compute cosine similarity
        similarity = torch.cosine_similarity(image_features, text_features).item()
        return similarity
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Analyze the relationship for each row in the dataset
similarity_scores = []
for index, row in df.iterrows():
    image_filename = row["Image Filename"]
    article_text = row["Full Article"]

    # Construct image path (ensure it matches your image storage location)
    image_path = f"politifact_images/{image_filename}"

    # Compute similarity score
    similarity = get_image_text_similarity(image_path, article_text)
    similarity_scores.append(similarity)

# Add similarity scores to dataset
df["Image-Article Similarity"] = similarity_scores

# Save updated dataset
df.to_csv("politifact_with_similarity.csv", index=False)

print("Image-Article similarity analysis complete. Dataset saved as politifact_with_similarity.csv.")
