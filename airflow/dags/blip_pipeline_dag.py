from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

class BlipForFakeNewsClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.blip = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.classifier = nn.Linear(self.blip.config.projection_dim, num_labels)

    def forward(self, input_ids, pixel_values, attention_mask=None):
        outputs = self.blip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        pooled_output = outputs.image_embeds
        return self.classifier(pooled_output)

def preprocess_data():
    df = pd.read_csv("data/politifact_with_local_images.csv")
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
    df.to_csv("data/processed_fake_news.csv", index=False)
    print("✅ Preprocessing done.")

def train_blip_model():
    df = pd.read_csv("data/processed_fake_news.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    train_ds = FakeNewsDataset(train_df)
    val_ds = FakeNewsDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlipForFakeNewsClassification(num_labels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
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

def evaluate_blip_model():
    df = pd.read_csv("data/processed_fake_news.csv")
    _, val_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    val_ds = FakeNewsDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

with DAG(
    dag_id="blip_mlops_all_in_one",
    default_args={'owner': 'airflow', 'start_date': datetime(2024, 1, 1)},
    schedule_interval=None,
    catchup=False,
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_blip_model
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_blip_model
    )

    preprocess >> train >> evaluate
