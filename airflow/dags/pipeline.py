from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mobile_item_rec_pipeline',
    default_args=default_args,
    description='Mobile Item Recommendation Pipeline',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'mobile', 'feast'],
) as dag:

    # 1. Data Preprocessing
    # Runs the script that generates the Train/Val/Test parquet files
    # AND the discrete feature tables (User, Item, UI, UC) for Feast.
    preprocess_task = BashOperator(
        task_id='generate_features',
        bash_command='python /app/src/preprocessing.py',
        doc_md="Generates Parquet feature files from raw data."
    )

    # 2. Update Feature Store (Feast)
    # This loads the newly generated Parquet files into Redis (Online Store).
    # 1. 'feast apply': Registers any new feature definitions.
    # 2. 'feast materialize-incremental': Syncs data from Offline -> Online.
    #    We use $(date -u) to sync everything up to the current moment.
    materialize_features_task = BashOperator(
        task_id='materialize_to_redis',
        bash_command="""
        cd /app/feature_repo && \
        feast apply && \
        feast materialize 2014-11-01T00:00:00 2015-01-01T00:00:00
        """,
        doc_md="Syncs offline parquet features to Redis for low-latency serving."
    )

    # 3. Model Training
    # Runs XGBoost training, Hyperparameter Tuning (Optuna), and Evaluation.
    # If the model is good (Global F1 > Threshold), it registers to MLflow.
    train_model_task = BashOperator(
        task_id='train_and_register_model',
        bash_command='python /app/src/train.py',
        env={
            # Ensure script knows where to find MLflow (Networked container)
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000'
        },
        doc_md="Trains XGBoost model, logs to MLflow, and promotes to Production if valid."
    )

    # Task Dependencies
    # Preprocessing must happen first.
    # Then we can run Materialization and Training (sequentially or parallel).
    # We do sequential here to be safe with resources.
    
    preprocess_task >> materialize_features_task >> train_model_task