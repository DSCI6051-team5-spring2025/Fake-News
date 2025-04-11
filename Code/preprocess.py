import pandas as pd
import os

def preprocess_data(input_csv="data/politifact_with_local_images.csv", output_csv="data/processed_fake_news.csv", image_folder="data/images"):
    df = pd.read_csv(input_csv)
    df['Image_Path'] = df['Image_Path'].str.replace('\\', '/', regex=True)
    df['Image_Path'] = df['Image_Path'].str.replace(r'^images[\/]', '', regex=True)
    df['image_path'] = df['Image_Path'].apply(lambda x: os.path.join(image_folder, x))
    df['text'] = df['Claim']

    real_labels = ['true', 'mostly true', 'half true']
    fake_labels = ['false', 'mostly false', 'pants on fire']

    def map_rating(rating):
        rating = str(rating).strip().lower()
        if rating in real_labels: return 1
        if rating in fake_labels: return 0
        return None

    df['binary_label'] = df['Rating'].apply(map_rating)
    df = df.dropna(subset=['binary_label'])
    df['binary_label'] = df['binary_label'].astype(int)

    df.to_csv(output_csv, index=False)
    print("✅ Preprocessing complete.")