import os
import requests
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# Load the cleaned dataset
df = pd.read_csv("politifact_cleaned_final.csv")

# Directory to save images
image_dir = "politifact_images"
os.makedirs(image_dir, exist_ok=True)

# Function to download an image with optimized handling
def download_image(index, image_url):
    filename = f"image_{index}.jpg"
    file_path = os.path.join(image_dir, filename)

    try:
        response = requests.get(image_url, stream=True, timeout=5)  # 5-sec timeout
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return filename  # Success
    except Exception as e:
        return "Error"  # Failed download

# Use multithreading to speed up downloads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(download_image, df.index, df["Image URL"]),
                        total=len(df), desc="Downloading Images"))

# Add image filenames to the dataset
df["Image Filename"] = results

# Remove failed downloads
df_final = df[df["Image Filename"] != "Error"].reset_index(drop=True)

# Save the updated dataset
df_final.to_csv("politifact_cleaned_with_images.csv", index=False)

print("✅ Image Download Complete!")
print("📂 Dataset saved as: politifact_cleaned_with_images.csv")
print("🖼️ Images saved in folder:", image_dir)
