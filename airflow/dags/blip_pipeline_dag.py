from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# Add script path
sys.path.append("C:/Users/rajar/OneDrive/Desktop/New folder/Fake-News/Code")

# Import tasks
from preprocess import preprocess_data
from train import train_blip_model
from evaluate import evaluate_blip_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 4, 1),
}

with DAG(
    dag_id="blip_mlops_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False
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
