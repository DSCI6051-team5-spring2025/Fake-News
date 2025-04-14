import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor
from model import BlipForFakeNewsClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        text = row["text"]
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        for k in inputs:
            inputs[k] = inputs[k].squeeze(0)
        inputs["binary_label"] = torch.tensor(row["binary_label"])
        return inputs

def evaluate_blip_model():
    df = pd.read_csv("data/processed_fake_news.csv")
    _, val_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    val_ds = FakeNewsDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = BlipForFakeNewsClassification(num_labels=2).to(device)
    model.load_state_dict(torch.load("blip_best_model.pth"))
    model.eval()

    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["binary_label"].to(device)

            logits = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(val_labels, val_preds)
    cm = confusion_matrix(val_labels, val_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Validation Accuracy: {acc:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    print("✅ Evaluation complete. Accuracy:", acc)