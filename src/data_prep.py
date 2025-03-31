def load_and_preprocess(csv_path, image_dir=None):
    import pandas as pd
    import os
    from PIL import Image
    import torchvision.transforms as transforms

    df = pd.read_csv(csv_path)

    processed_data = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for _, row in df.iterrows():
        text = row['Claim']                         # <-- updated
        image_path = row['Image URL']               # <-- updated
        label = 1 if row['Rating'].strip().upper() == "TRUE" else 0  # <-- updated

        if image_path and image_path != "N/A":
            try:
                filename = os.path.basename(image_path)
                image = Image.open(os.path.join("data/raw/images", filename)).convert("RGB")
            except Exception as e:
                print(f"Error loading image: {image_path}, using placeholder. {e}")
                image = Image.new("RGB", (224, 224), color="white")
        else:
            image = Image.new("RGB", (224, 224), color="white")

        image = transform(image)

        processed_data.append({
            'text': text,
            'image': image,
            'label': label
        })

    return processed_data
