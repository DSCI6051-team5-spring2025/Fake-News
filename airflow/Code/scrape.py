import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import os

def update_politifact_dataset(csv_path="data/politifact_with_local_images.csv", limit=20):
    base_url = "https://www.politifact.com"
    list_url = f"{base_url}/factchecks/list/"

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        existing_urls = set(df_existing["Article URL"].dropna().tolist())
    else:
        df_existing = pd.DataFrame()
        existing_urls = set()

    response = requests.get(list_url)
    soup = BeautifulSoup(response.content, "html.parser")

    new_entries = []
    items = soup.select("li.o-listicle__item")[:limit]
    for item in items:
        link_tag = item.select_one("a.m-statement__link")
        if not link_tag:
            continue
        article_url = urljoin(base_url, link_tag.get("href"))
        if article_url in existing_urls:
            continue
        claim = link_tag.text.strip()
        rating_tag = item.select_one("div.m-statement__meter img")
        rating = rating_tag["alt"].strip() if rating_tag else "Not Rated"
        img_tag = item.select_one("img.c-image")
        image_url = img_tag["src"] if img_tag else ""
        new_entries.append({
            "Article URL": article_url,
            "Claim": claim,
            "Rating": rating,
            "Image URL": image_url,
            "Image_Path": ""
        })

    if new_entries:
        df_new = pd.DataFrame(new_entries)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"✅ Added {len(df_new)} new articles.")
    else:
        print("ℹ️ No new articles found.")