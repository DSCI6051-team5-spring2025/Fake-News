from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from scripts.scrape import update_politifact_dataset
from scripts.preprocess import preprocess_data
from scripts.train import train_blip_model
from scripts.evaluate import evaluate_blip_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 4, 1),
}

with DAG("blip_mlops_pipeline", default_args=default_args, schedule_interval=None, catchup=False) as dag:
    fetch_data = PythonOperator(task_id="fetch_new_data", python_callable=update_politifact_dataset)
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    train = PythonOperator(task_id="train_model", python_callable=train_blip_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_blip_model)

    fetch_data >> preprocess >> train >> evaluate
