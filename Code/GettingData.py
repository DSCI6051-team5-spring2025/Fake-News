import os
import requests
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Base URL for PolitiFact fact-checks
BASE_URL = "https://www.politifact.com/factchecks/list/?page="

# Number of pages to scrape
NUM_PAGES = 18  

# Directory to save images
IMAGE_DIR = "politifact_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Step 1: Extract Article Details (WITHOUT IMAGES & RATINGS)
data = []
fact_check_urls = []

for page in range(1, NUM_PAGES + 1):
    url = BASE_URL + str(page)
    driver.get(url)
    time.sleep(3)  # Wait for content to load

    articles = driver.find_elements(By.CLASS_NAME, "o-listicle__item")

    for article in articles:
        try:
            claim_element = article.find_element(By.CLASS_NAME, "m-statement__quote")
            claim_text = claim_element.text.strip()
            claim_url = claim_element.find_element(By.TAG_NAME, "a").get_attribute("href")
            fact_check_urls.append((claim_url, claim_text))
        except Exception as e:
            print(f"Skipping entry due to error: {e}")

# Step 2: Visit each fact-check page separately
for claim_url, claim_text in fact_check_urls:
    try:
        driver.get(claim_url)
        time.sleep(3)

        try:
            claimant = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "m-statement__meta"))
            ).text.strip()
        except:
            claimant = "N/A"

        try:
            date = driver.find_element(By.CLASS_NAME, "m-statement__desc").text.strip()
        except:
            date = "N/A"

        try:
            article_text = driver.find_element(By.CLASS_NAME, "m-textblock").text.strip()
        except:
            article_text = "N/A"

        try:
            source_element = driver.find_element(By.CLASS_NAME, "m-source__content")
            source_text = source_element.text.strip()
        except:
            source_text = "N/A"

        # Store initial dataset (without images & ratings)
        data.append({
            "Claimant": claimant,
            "Claim": claim_text,
            "Date": date,
            "URL": claim_url,
            "Source": source_text,
            "Full Article": article_text,
            "Rating": "N/A",  # Placeholder for now
            "Image Filename": "N/A"
        })
    except Exception as e:
        print(f"Skipping article due to error: {e}")

# Close Selenium WebDriver
driver.quit()

# Convert to DataFrame and Save to CSV
df = pd.DataFrame(data)
df.to_csv("politifact_articles.csv", index=False)

# Step 3: Extract Ratings Using BeautifulSoup
def extract_rating(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        rating_element = soup.find("div", class_="m-statement__meter")
        if rating_element:
            rating_img = rating_element.find("img")
            if rating_img and "alt" in rating_img.attrs:
                return rating_img["alt"].strip()

    return "N/A"

# Update dataset with correct ratings
df["Rating"] = df["URL"].apply(extract_rating)

# Step 4: Extract Images Using BeautifulSoup
def extract_images_from_article(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        images = soup.find_all("img")
        for img in images:
            img_url = img.get("src")
            if img_url and "CACHE/images/politifact/photos" in img_url:
                return img_url  # Return the first relevant image
    return None

# Download Images and Update Dataset
for index, row in df.iterrows():
    article_url = row["URL"]
    image_url = extract_images_from_article(article_url)

    if image_url:
        image_filename = image_url.split("/")[-1]
        image_path = os.path.join(IMAGE_DIR, image_filename)

        # Download image
        img_data = requests.get(image_url).content
        with open(image_path, "wb") as img_file:
            img_file.write(img_data)

        # Store the image filename in the dataset
        df.at[index, "Image Filename"] = image_filename
    else:
        print(f"No image found for: {article_url}")

# Step 5: Save Final Dataset with Correct Ratings & Images
df.to_csv("politifact_final_dataset.csv", index=False)

print(f"Scraping complete. {len(df)} fact-checks saved to politifact_final_dataset.csv.")
print(f"Images saved in: {IMAGE_DIR}")
