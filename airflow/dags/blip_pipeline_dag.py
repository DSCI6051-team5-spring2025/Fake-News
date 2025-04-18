from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pandas as pd
from urllib.parse import urlparse
import hashlib
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import urlparse


def get_webdriver():
    options = Options()
    options.add_argument('--ignore-ssl-errors=yes')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')

    # Custom user agent
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')

    # Connect to remote ChromeDriver
    driver = webdriver.Remote(
        command_executor='http://remote_chromedriver:4444/wd/hub',
        options=options
    )
    return driver




class FakeNewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.data.iloc[idx]

        image = Image.open(row["Image_Path"]).convert("RGB").resize((224, 224), resample=Image.Resampling.BILINEAR)

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

def scrape_politifact():
    def get_headers():
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

    def extract_image_url(article_url):
        try:
            for _ in range(3):
                response = requests.get(article_url, headers=get_headers(), timeout=10)
                if response.status_code == 200:
                    break
                time.sleep(5)
            if response.status_code != 200:
                return "N/A"
            soup = BeautifulSoup(response.text, "html.parser")
            main_section = soup.find("div", class_="t-row__center")
            if not main_section:
                return "N/A"
            article = main_section.find("article", class_="m-display__inner")
            if not article:
                return "N/A"
            picture_tag = article.find("picture")
            if picture_tag:
                source_tag = picture_tag.find("source", {"media": "(min-width: 520px)"})
                if source_tag and "srcset" in source_tag.attrs:
                    return source_tag["srcset"]
                source_tag = picture_tag.find("source")
                if source_tag and "srcset" in source_tag.attrs:
                    return source_tag["srcset"]
                img_tag = picture_tag.find("img")
                if img_tag and "src" in img_tag.attrs:
                    return img_tag["src"]
            return "N/A"
        except Exception:
            return "N/A"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = get_webdriver()
    driver.get("https://www.politifact.com/factchecks/list/")
    time.sleep(5)

    articles = driver.find_elements(By.CLASS_NAME, "o-listicle__item")
    extracted = []

    for article in articles:
        try:
            quote_el = article.find_element(By.CLASS_NAME, "m-statement__quote")
            claim = quote_el.text.strip()
            article_url = quote_el.find_element(By.TAG_NAME, "a").get_attribute("href")
            rating = article.find_element(By.CLASS_NAME, "m-statement__meter").find_element(By.TAG_NAME, "img").get_attribute("alt")
            extracted.append({"Claim": claim, "Rating": rating, "Article URL": article_url})
        except Exception:
            continue

    driver.quit()
    for item in extracted:
        item["Image URL"] = extract_image_url(item["Article URL"])
    df = pd.DataFrame(extracted)
    df.drop_duplicates(subset=["Claim", "Article URL"], inplace=True)
    df = df[df["Image URL"] != "N/A"].dropna(subset=["Image URL"]).reset_index(drop=True)
    df.to_pickle("/opt/airflow/data/temp_scraped_df.pkl")
    print(" Scraped and stored DataFrame in pickle.")


def hash_filename(url):
    """Generate a unique filename using MD5 hash of the URL"""
    return hashlib.md5(url.encode()).hexdigest() + os.path.splitext(urlparse(url).path)[-1] or ".jpg"

def merge_and_download():
    df_new = pd.read_pickle("/opt/airflow/data/temp_scraped_df.pkl")
    csv_path = "/opt/airflow/data/raw/politifact_with_local_images.csv"
    image_dir = "/opt/airflow/data/raw/images"

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing.drop_duplicates(subset=["Claim", "Article URL"], inplace=True)
    else:
        df_existing = pd.DataFrame(columns=df_new.columns)

    df_new.drop_duplicates(subset=["Claim", "Article URL"], inplace=True)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=["Claim", "Article URL"], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    os.makedirs(image_dir, exist_ok=True)
    image_paths = []

    for _, row in df_combined.iterrows():
        img_url = row.get("Image URL", "")
        if not img_url or img_url == "N/A":
            image_paths.append("N/A")
            continue

        try:
            filename = hash_filename(img_url)
            full_path = os.path.join("/opt/airflow/data/raw/images", filename).replace("\\", "/")


            if not os.path.exists(full_path):
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(full_path, "wb") as f:
                        f.write(response.content)
                else:
                    full_path = "N/A"

            image_paths.append(full_path if os.path.exists(full_path) else "N/A")

        except Exception:
            image_paths.append("N/A")

    df_combined["Image_Path"] = image_paths
    df_combined = df_combined[df_combined["Image_Path"] != "N/A"]
    df_combined.to_csv(csv_path, index=False)
    print(f"✅ Merged, downloaded images, and saved final dataset. Total: {len(df_combined)}")

def preprocess_data():
    csv_path = "/opt/airflow/data/raw/politifact_with_local_images.csv"
    df = pd.read_csv(csv_path)

    # Clean and map labels
    df["text"] = df["Claim"]
    real = ["true", "mostly true", "half true"]
    fake = ["false", "mostly false", "pants on fire"]

    def map_label(rating):
        rating = str(rating).strip().lower()
        if rating in real:
            return 1
        if rating in fake:
            return 0
        return None

    df["binary_label"] = df["Rating"].apply(map_label)
    df = df.dropna(subset=["binary_label"])
    df["binary_label"] = df["binary_label"].astype(int)

    # Normalize and fix image paths
    df["Image_Path"] = df["Image_Path"].astype(str).str.replace("\\", "/", regex=False)
    df["Image_Path"] = df["Image_Path"].str.replace("/opt/airflow/images", "/opt/airflow/data/raw/images")

    # Filter out rows without required fields
    df = df[df["Image_Path"].notna()]
    df = df[df["text"].notna()]
    print("✅ Sample image paths before saving:")
    print(df["Image_Path"].head(10).to_list())


    # Save cleaned data
    df.to_csv("/opt/airflow/data/processed/processed_fake_news.csv", index=False)
    print(f"✅ Preprocessing complete. Rows: {len(df)}")


def train_blip_model():
    data_path = "/opt/airflow/data/processed/processed_fake_news.csv"
    model_path = "/opt/airflow/data/model/blip_best_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df = pd.read_csv(data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    train_ds = FakeNewsDataset(train_df)
    val_ds = FakeNewsDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlipForFakeNewsClassification(num_labels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    for epoch in range(1):
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
            total_loss += loss.item()

        train_acc = accuracy_score(all_labels, all_preds)
        print(f"✅ Epoch {epoch+1}: Train Accuracy = {train_acc:.4f} | Loss = {total_loss:.4f}")
        if train_acc > best_val_acc:
            best_val_acc = train_acc
            torch.save(model.state_dict(), model_path)
            print("📌 Best model saved!")

def evaluate_blip_model():
    data_path = "/opt/airflow/data/processed/processed_fake_news.csv"
    model_path = "/opt/airflow/data/model/blip_best_model.pth"
    output_path = "/opt/airflow/data/model/confusion_matrix.png"

    df = pd.read_csv(data_path)
    _, val_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    val_ds = FakeNewsDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlipForFakeNewsClassification(num_labels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print("✅ Evaluation complete. Accuracy:", acc)

with DAG(
    dag_id="blip_mlops_all_in_one",
    default_args={'owner': 'airflow', 'start_date': datetime(2024, 1, 1)},
    schedule_interval=None,
    catchup=False,
) as dag:
    
    task_scrape = PythonOperator(
        task_id="scrape_data",
        python_callable=scrape_politifact
    )

    task_merge = PythonOperator(
        task_id="merge_and_download_images",
        python_callable=merge_and_download
    )

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

    task_scrape >> task_merge >> preprocess >> train >> evaluate
