import gradio as gr
import os
import pandas as pd
from sqlalchemy import create_engine

def trigger_airflow_dag():
    os.system("airflow dags trigger blip_mlops_all_in_one")
    return "✅ DAG triggered successfully."

def fetch_task_status():
    try:
        engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres:5432/airflow')
        df = pd.read_sql("""
            SELECT dag_id, task_id, state, start_date, end_date
            FROM task_instance
            WHERE dag_id = 'blip_mlops_all_in_one'
            ORDER BY execution_date DESC, task_id
            LIMIT 10;
        """, engine)
        return df.to_markdown(index=False)
    except Exception as e:
        return f"⚠️ Error fetching task status: {e}"

def view_outputs():
    try:
        report = ""
        if os.path.exists("/opt/airflow/data/temp_scraped_df.pkl"):
            df_scraped = pd.read_pickle("/opt/airflow/data/temp_scraped_df.pkl")
            report += f"📰 Scraped: {len(df_scraped)} claims\n"

        if os.path.exists("/opt/airflow/data/raw/politifact_with_local_images.csv"):
            df_merged = pd.read_csv("/opt/airflow/data/raw/politifact_with_local_images.csv")
            report += f"🖼️ Merged: {len(df_merged)} records with image paths\n"

        if os.path.exists("/opt/airflow/data/processed/processed_fake_news.csv"):
            df_cleaned = pd.read_csv("/opt/airflow/data/processed/processed_fake_news.csv")
            report += f"🧹 Cleaned: {len(df_cleaned)} processed records\n"

        if os.path.exists("/opt/airflow/data/model/confusion_matrix.png"):
            report += "📊 Confusion matrix generated.\n"

        return report or "📂 No output data found yet."
    except Exception as e:
        return f"⚠️ Error reading outputs: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Fake News MLOps Dashboard")

    with gr.Row():
        trigger_btn = gr.Button("▶️ Trigger Pipeline")
        status_btn = gr.Button("📋 Task Status")
        output_btn = gr.Button("📦 View Output Summary")

    output_box = gr.Textbox(label="Live Pipeline Info", lines=10)

    trigger_btn.click(fn=trigger_airflow_dag, outputs=output_box)
    status_btn.click(fn=fetch_task_status, outputs=output_box)
    output_btn.click(fn=view_outputs, outputs=output_box)

# Run on port 8050 internally (if using Docker)
demo.launch(server_name="0.0.0.0", server_port=8050, share=False)
