import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import timedelta, datetime
import os
import logging
import optuna
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# Use absolute paths for Docker, relative paths for local dev
DATA_PATH = os.getenv("DATA_PATH", "/app/data/raw")
FEATURE_PATH = os.getenv("FEATURE_PATH", "/app/data/features")
MLFLOW_EXPERIMENT_NAME = "mobile-item-recommendation-experiment"
RAW_USERS = f"{DATA_PATH}/users.parquet"

def downsample(df, ratio=20):
    positives = df[df['label'] == 1]
    negatives = df[df['label'] == 0]
    
    if len(negatives) > len(positives) * ratio:
        negatives = negatives.sample(n=len(positives) * ratio, random_state=42)
    
    return pd.concat([positives, negatives]).sample(frac=1)

def build_dataset_from_features(target_date_str, raw_users_path):

    target_date = pd.Timestamp(target_date_str).date()
    logger.info(f"--- Building Dataset for Target Date: {target_date} ---")

    df = pd.read_parquet(raw_users_path)
    df['date'] = pd.to_datetime(df['time']).dt.date

    history_df = df[df['date'] < target_date]
    target_df = df[df['date'] == target_date]

    start_anchor = target_date - timedelta(days=1)
    candidates = history_df[history_df['date'] >= start_anchor][['user_id', 'item_id', 'item_category']].drop_duplicates()

    buys = target_df[target_df['behavior_type'] == 4]
    positive_pairs = set(zip(buys['user_id'], buys['item_id']))
    candidates['label'] = candidates.apply(lambda x: 1 if (x['user_id'], x['item_id']) in positive_pairs else 0, axis=1)

    windows = [1, 3]
    for days in windows:
        prefix = f"last_{days}d"
        timestamp_str = str(target_date) # e.g. 2014-12-18
        
        # Load Tables
        try:
            logger.info(f"Loading {prefix} features for {timestamp_str}...")

            item_df = pd.read_parquet(f"{FEATURE_PATH}/item_stats_{prefix}_{timestamp_str}.parquet")
            user_df = pd.read_parquet(f"{FEATURE_PATH}/user_stats_{prefix}_{timestamp_str}.parquet")
            ui_df = pd.read_parquet(f"{FEATURE_PATH}/ui_stats_{prefix}_{timestamp_str}.parquet")
            uc_df = pd.read_parquet(f"{FEATURE_PATH}/uc_stats_{prefix}_{timestamp_str}.parquet")
            
            # Drop timestamp columns to avoid duplicates during merge
            item_df = item_df.drop(columns=['event_timestamp'], errors='ignore')
            user_df = user_df.drop(columns=['event_timestamp'], errors='ignore')
            ui_df = ui_df.drop(columns=['event_timestamp'], errors='ignore')
            uc_df = uc_df.drop(columns=['event_timestamp'], errors='ignore')

            # Merge
            candidates = candidates.merge(item_df, on='item_id', how='left').fillna(0)
            candidates = candidates.merge(user_df, on='user_id', how='left').fillna(0)
            
            # Create composite keys for merging (matching preprocessing.py format)
            candidates['user_item_id'] = candidates['user_id'].astype(str) + '_' + candidates['item_id'].astype(str)
            candidates['user_category_id'] = candidates['user_id'].astype(str) + '_' + candidates['item_category'].astype(str)
            
            candidates = candidates.merge(ui_df, on='user_item_id', how='left').fillna(0)
            candidates = candidates.merge(uc_df, on='user_category_id', how='left').fillna(0)
            
            # Drop composite keys after merge (they're not needed for training)
            candidates = candidates.drop(columns=['user_item_id', 'user_category_id'], errors='ignore')
            
        except FileNotFoundError as e:
            logger.warning(f"Missing feature file for {prefix}: {e}")

    # Join Recency Features
    try:
        logger.info(f"Loading Recency features for {target_date}...")
        recency_df = pd.read_parquet(f"{FEATURE_PATH}/recency_stats_{target_date}.parquet")
        recency_df = recency_df.drop(columns=['event_timestamp'], errors='ignore')
        # Create composite key for merging
        candidates['user_item_id'] = candidates['user_id'].astype(str) + '_' + candidates['item_id'].astype(str)
        candidates = candidates.merge(recency_df, on='user_item_id', how='left').fillna(-1)
        candidates = candidates.drop(columns=['user_item_id'], errors='ignore')
    except FileNotFoundError:
        logger.warning("Missing Recency features")

    logger.info(f"Dataset shape: {candidates.shape}")
    return candidates


def evaluate_model(model, dataset_df, feature_cols, total_daily_buys=None, dataset_name="Validation"):

    preds = model.predict(dataset_df[feature_cols])
    pred_indices = (preds == 1)
    prediction_pairs = set(zip(dataset_df.loc[pred_indices, 'user_id'], dataset_df.loc[pred_indices, 'item_id']))

    # Actual Positives IN CANDIDATE SET: (User, Item) where label was 1
    actual_indices = (dataset_df['label'] == 1)
    reference_pairs = set(zip(dataset_df.loc[actual_indices, 'user_id'], dataset_df.loc[actual_indices, 'item_id']))
    # Intersection = True Positives found by the model
    intersection = prediction_pairs & reference_pairs
    tp_count = len(intersection)
    # Precision is always (TP / Predicted)
    precision = tp_count / len(prediction_pairs) if prediction_pairs else 0.0
    # Recall Logic
    if total_daily_buys:
        # GLOBAL RECALL: TP / All purchases in the world that day
        recall = tp_count / total_daily_buys
        metric_type = "Global"
    else:
        # LOCAL RECALL: TP / All purchases in our candidate set
        recall = tp_count / len(reference_pairs) if reference_pairs else 0.0
        metric_type = "Local"
    # F1 Calculation
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    logger.info(f"[{dataset_name} - {metric_type}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1

model_params = {
    "LogisticRegression": {
        "class_weight": 'balanced',
        "max_iter": 2000,
        "random_state": 42
    },
    "RandomForest": {
        "n_estimators": 50,
        "max_depth": 10,
        "random_state": 42
    },
    "XGBoost": {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "eval_metric": 'logloss',
        "random_state": 42
    }
}

def train_and_evaluate_model():
    
    # Use environment variable if set, otherwise default to localhost (for local dev)
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Building Training Dataset...")
    train_df = build_dataset_from_features("2014-12-06", RAW_USERS)
    logger.info("Building Validation Dataset...")
    val_df = build_dataset_from_features("2014-12-12", RAW_USERS)
    logger.info("Building Test Dataset...")
    test_df = build_dataset_from_features("2014-12-18", RAW_USERS)

    # Calculate Global Truth Counts for Evaluation logic
    raw_df = pd.read_parquet(f"{DATA_PATH}/users.parquet")
    val_global_buys = len(raw_df[(raw_df['time'].dt.date == pd.Timestamp("2014-12-12").date()) & (raw_df['behavior_type'] == 4)])
    test_global_buys = len(raw_df[(raw_df['time'].dt.date == pd.Timestamp("2014-12-18").date()) & (raw_df['behavior_type'] == 4)])

    drop_cols = ['user_id', 'item_id', 'item_category', 'label']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    train_balanced = downsample(train_df, ratio=20)
    X_train = train_balanced[feature_cols]
    y_train = train_balanced['label']
    X_val = val_df[feature_cols]
    y_val = val_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']

    signature = infer_signature(X_train, y_train)
    input_example = X_train.head(1)

    # Generate a unique timestamp for this training session to avoid duplicate runs
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run Baselines
    best_model_info = {"f1": -1, "run_id": None, "description": None}

    def run_model(model_name, model_obj, params):
        nonlocal best_model_info
        # Append timestamp to make run names unique
        unique_run_name = f"{model_name}_{session_timestamp}"
        with mlflow.start_run(run_name=unique_run_name):
            logger.info(f"Training {model_name}...")
            model_obj.fit(X_train, y_train)
            
            # 1. Local Evaluation (Candidate Set)
            logger.info("Evaluating Local...")
            p_local, r_local, f1_local = evaluate_model(
                model_obj, 
                val_df, 
                feature_cols, 
                total_daily_buys=None, 
                dataset_name="Val_Local")
            # 2. Global Evaluation (Business Metric)
            logger.info("Evaluating Global...")
            p_global, r_global, f1_global = evaluate_model(
                model_obj, 
                val_df,
                feature_cols, 
                total_daily_buys=val_global_buys, 
                dataset_name="Val_Global")
            
            logger.info("Logging Metrics...")
            mlflow.log_metrics({
                "val_f1_local": f1_local,
                "val_precision_local": p_local,
                "val_recall_local": r_local,
                "val_f1_global": f1_global, 
                "val_recall_global": r_global
            })
            logger.info("Logging Params...")
            mlflow.log_params({
                "model_type": model_name.split('_')[1], # e.g. "LogisticRegression"
                **params
            })
            logger.info("Logging Model...")
            mlflow.sklearn.log_model(model_obj, name="model", signature=signature, input_example=input_example)

            # Use Local F1 for model selection (since Global F1 is just a scaled version of Local F1 for the same candidate set)
            if f1_local > best_model_info["f1"]:
                logger.info("Updating Best Model Info...")
                active_run = mlflow.active_run()
                best_model_info = {
                    "f1": f1_local,
                    "run_id": active_run.info.run_id if active_run else None,
                    "description": model_name
                }

    # Run Baselines
    logger.info("Running Baselines...")
    run_model("baseline_logistic_regression", LogisticRegression(**model_params["LogisticRegression"]), model_params["LogisticRegression"])
    run_model("baseline_random_forest", RandomForestClassifier(**model_params["RandomForest"]), model_params["RandomForest"])
    run_model("baseline_xgboost", xgb.XGBClassifier(**model_params["XGBoost"]), model_params["XGBoost"])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 10),
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Score on Validation Set
        preds = model.predict(X_val)
        return f1_score(y_val, preds)

    logger.info("Starting Optuna Tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    logger.info(f"Best Val F1: {study.best_value:.4f}") 
    
    best_params = {
        **study.best_params,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False
    }

    # Append timestamp to make run name unique
    unique_tuned_run_name = f"xgboost_optuna_tuned_{session_timestamp}"
    with mlflow.start_run(run_name=unique_tuned_run_name):
        logger.info("Training Tuned Model...")
        tuned_model = xgb.XGBClassifier(**best_params)
        tuned_model.fit(X_train, y_train)

        # Validation Metrics (Local & Global)
        logger.info("Evaluating Validation Local...")
        p_val_local, r_val_local, f1_val_local = evaluate_model(tuned_model, val_df, feature_cols, total_daily_buys=None, dataset_name="Val_Local")
        logger.info("Evaluating Validation Global...")
        p_val_global, r_val_global, f1_val_global = evaluate_model(tuned_model, val_df, feature_cols, total_daily_buys=val_global_buys, dataset_name="Val_Global")
        
        # Test Metrics (Local & Global)
        logger.info("Evaluating Test Local...")
        p_test_local, r_test_local, f1_test_local = evaluate_model(tuned_model, test_df, feature_cols, total_daily_buys=None, dataset_name="Test_Local")
        logger.info("Evaluating Test Global...")
        p_test_global, r_test_global, f1_test_global = evaluate_model(tuned_model, test_df, feature_cols, total_daily_buys=test_global_buys, dataset_name="Test_Global")

        logger.info("Logging Params...")
        mlflow.log_params({
            "model_type": "XGBoost_Optuna",
            **best_params
        })
        logger.info("Logging Metrics...")
        mlflow.log_metrics({
            "val_f1_local": f1_val_local,
            "val_f1_global": f1_val_global,
            "test_f1_local": f1_test_local,
            "test_f1_global": f1_test_global,
            "best_trial": study.best_trial.number
        })
        logger.info("Logging Model...")
        mlflow.sklearn.log_model(tuned_model, name="model", signature=signature, input_example=input_example)

        if f1_val_local > best_model_info["f1"]:
            logger.info("Updating Best Model Info...")
            active_run = mlflow.active_run()
            best_model_info = {
                "f1": f1_val_local,
                "run_id": active_run.info.run_id if active_run else None,
                "description": "xgboost_optuna_tuned"
            }

    if best_model_info["run_id"]:
        logger.info(f"Registering best model ({best_model_info['description']}) with val F1={best_model_info['f1']:.4f}")
        model = mlflow.register_model(f"runs:/{best_model_info['run_id']}/model", "mobile_item_rec_model")
        client = MlflowClient()
        
        # Use aliases instead of deprecated stages (MLflow 2.9+)
        # Set "champion" alias to point to this version
        try:
            client.set_registered_model_alias(
                name="mobile_item_rec_model",
                alias="champion",
                version=model.version
            )
            logger.info(f"Set 'champion' alias to version {model.version}")
        except Exception as e:
            logger.warning(f"Could not set alias (may not be supported in this MLflow version): {e}")
            # Fallback to stage transition for older MLflow versions
            try:
                client.transition_model_version_stage(
                    name="mobile_item_rec_model",
                    version=model.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(f"Promoted version {model.version} to Production Stage")
            except Exception as e2:
                logger.warning(f"Could not transition stage: {e2}")
    else:
        logger.warning("No model run recorded for registration.")

if __name__ == "__main__":
    train_and_evaluate_model()