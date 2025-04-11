import torch
import torch.nn as nn
import pandas as pd
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor
from model import BlipForFakeNewsClassification
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        from PIL import Image
        image = Image.open(row["image_path"]).convert("RGB")
        text = row["text"]
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        for k in inputs:
            inputs[k] = inputs[k].squeeze(0)
        inputs["binary_label"] = torch.tensor(row["binary_label"])
        return inputs

def train_blip_model():
    df = pd.read_csv("data/processed_fake_news.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    train_ds = FakeNewsDataset(train_df)
    val_ds = FakeNewsDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = BlipForFakeNewsClassification(num_labels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("blip-fake-news")

    with mlflow.start_run():
        best_val_acc = 0
        for epoch in range(3):
            model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["binary_label"].to(device)

                logits = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(all_labels, all_preds)
            print(f"Epoch {epoch+1}: Train Accuracy = {train_acc:.4f}")
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                torch.save(model.state_dict(), "blip_best_model.pth")

        mlflow.log_metric("final_train_accuracy", train_acc)
        mlflow.pytorch.log_model(model, "blip_model")
        print("✅ Training complete and model logged to MLflow.")