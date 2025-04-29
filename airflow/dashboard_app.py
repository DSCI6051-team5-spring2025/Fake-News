import gradio as gr
import requests
import pandas as pd
import psycopg2
import os
from datetime import datetime
from gradio_client import Client, handle_file

# === CONFIG ===
DAG_ID = "blip_mlops_all_in_one"
AIRFLOW_TRIGGER_URL = f"http://airflow-webserver:8080/api/v1/dags/{DAG_ID}/dagRuns"
AIRFLOW_AUTH = ("airflow", "airflow")
DB_CONFIG = {
    "dbname": "airflow",
    "user": "airflow",
    "password": "airflow",
    "host": "postgres",
    "port": 5432
}
HF_SPACE_ID = "raja1729d/blip-fake-news-classifier"
LOG_PATH_TEMPLATE = "/opt/airflow/logs/dag_id={}/run_id={}/task_id={}/attempt=1.log"

# === STATE ===
last_run_id = ""

# === DASHBOARD FUNCTIONS ===
def trigger_pipeline():
    global last_run_id
    run_id = f"dashboard_trigger_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    response = requests.post(
        AIRFLOW_TRIGGER_URL,
        auth=AIRFLOW_AUTH,
        json={"conf": {}, "dag_run_id": run_id},
    )
    if response.status_code == 200:
        last_run_id = run_id
        return f"✅ DAG triggered! Run ID: `{run_id}`"
    else:
        return f"❌ Trigger failed:\n{response.text}"

def fetch_task_status(dag_id):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""SELECT run_id FROM dag_run WHERE dag_id = %s ORDER BY execution_date DESC LIMIT 1""", (dag_id,))
    run = cur.fetchone()
    if not run:
        return pd.DataFrame(columns=["task_id", "state", "start_date", "end_date"])
    run_id = run[0]
    cur.execute("""SELECT task_id, state, start_date, end_date FROM task_instance WHERE dag_id = %s AND run_id = %s ORDER BY start_date""", (dag_id, run_id))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=["task_id", "state", "start_date", "end_date"])

def fetch_task_output():
    global last_run_id
    if not last_run_id:
        return "No recent run_id found."
    task_outputs = []
    for task in ["scrape_data", "merge_and_download_images", "preprocess_data", "train_model", "evaluate_model", "deploy_to_hf"]:
        log_path = LOG_PATH_TEMPLATE.format(DAG_ID, last_run_id, task)
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                task_outputs.append(f"### 📄 {task}\n```\n{f.read()[-1500:]}\n```")
        else:
            task_outputs.append(f"### 📄 {task}\n⚠️ Log file not found.")
    return "\n\n".join(task_outputs)

def refresh_dashboard():
    status = fetch_task_status(DAG_ID)
    output = fetch_task_output()
    return status, output

# === HF MODEL INFERENCE ===
def predict_with_hf(image, text):
    try:
        client = Client(HF_SPACE_ID)
        result = client.predict(
            image=handle_file(image),
            text=text,
            api_name="/predict"
        )
        return f"🧠 Prediction: **{result}**"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tab("📊 MLOps Dashboard"):
        gr.Markdown("## 🧠 Fake News Detection - MLOps Dashboard")

        with gr.Row():
            trigger_btn = gr.Button("🚀 Trigger DAG")
            refresh_btn = gr.Button("🔄 Refresh Status")
            trigger_result = gr.Textbox(label="Trigger Result", lines=3)

        status_output = gr.Dataframe(headers=["task_id", "state", "start_date", "end_date"], label="📋 Task Status", interactive=False)
        output_box = gr.Textbox(
            label="🔍 Task Outputs",
            lines=200,         # height (more lines = more height)
            max_lines=5000,   # allow very big logs internally
            show_copy_button=True,  # allow user to easily copy
            interactive=False,      # disable editing
            autoscroll=True         # auto scroll to latest part
        )


        trigger_btn.click(fn=trigger_pipeline, outputs=trigger_result)
        refresh_btn.click(fn=refresh_dashboard, outputs=[status_output, output_box])

    with gr.Tab("🤖 Model Test (via Hugging Face)"):
        gr.Markdown("## 🧪 Try Deployed Fake News Classifier")

        test_image = gr.Image(type="filepath", label="News Image")
        test_text = gr.Textbox(label="News Text", placeholder="Enter the news claim...")
        predict_btn = gr.Button("🔍 Predict via HF Space")
        pred_output = gr.Markdown(label="Result")

        predict_btn.click(fn=predict_with_hf, inputs=[test_image, test_text], outputs=pred_output)

demo.launch(server_name="0.0.0.0", server_port=8051)
