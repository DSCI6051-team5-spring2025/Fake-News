# Start from the official Apache Airflow image
FROM apache/airflow:2.7.3

# Switch to root to install dependencies
USER root

# Install basic utilities
RUN apt-get update && apt-get install -y \
    wget curl unzip gnupg libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 \
    libappindicator3-1 libasound2 libatk-bridge2.0-0 libnspr4 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 libxtst6 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install Python packages
RUN pip install selenium beautifulsoup4 pandas scikit-learn gradio
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
