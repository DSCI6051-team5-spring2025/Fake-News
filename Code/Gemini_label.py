import google.generativeai as genai
import pandas as pd
import time
import os

# Configure Gemini API Key
GEMINI_API_KEY = "AIzaSyDHUT0g_9s_oIuqk6H5lvP-hzR6yPJ683s"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Load the cleaned dataset
df = pd.read_csv(r"C:\Users\rajar\OneDrive\Desktop\New folder\Fake-News\politifact_cleaned_dataset.csv")

# Define the directory where images are stored
IMAGE_DIR = r"C:\Users\rajar\OneDrive\Desktop\New folder\Fake-News\politifact_images"

# Function to analyze claim and image relationship using Gemini AI
def analyze_claim_image_relation(claim, full_article, image_filename):
    image_path = os.path.join(IMAGE_DIR, image_filename) if pd.notna(image_filename) else "No Image"

    prompt = f"""
    Analyze the following **fact-checked news article** and determine:
    
    1. **Is the image relevant to the claim?** (Answer: "Yes" or "No")
    2. **Does the article provide strong evidence for the claim?** (Answer: "Strong Evidence" or "Weak Evidence")

    **Fact-Checked Claim:** {claim}
    **Full Article Summary:** {full_article}

    **Image Path:** {image_path}

    Return only this format:
    - Image Relevance: Yes/No
    - Evidence Strength: Strong Evidence/Weak Evidence
    """

    try:
        model = genai.GenerativeModel('gemini-pro-latest')  # Use updated model name
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error processing claim: {e}")
        return "Unknown\nUnknown"

# Apply Gemini analysis
for index, row in df.iterrows():
    result = analyze_claim_image_relation(row["Claim"], row["Full Article"], row["Image Filename"])

    # Parse results
    results = result.split("\n")
    df.at[index, "Image_Relevance"] = results[0].split(":")[-1].strip() if len(results) > 0 else "Unknown"
    df.at[index, "Evidence_Strength"] = results[1].split(":")[-1].strip() if len(results) > 1 else "Unknown"

    time.sleep(1)  # Prevent API rate limits

# Save labeled dataset
output_file_path = r"C:\Users\rajar\OneDrive\Desktop\New folder\Fake-News\politifact_labeled_with_gemini.csv"
df.to_csv(output_file_path, index=False)

print(f" Analysis complete! Labeled dataset saved to: {output_file_path}")
