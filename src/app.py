import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import os
import asyncio
from datetime import timedelta

# Configuration
FEATURE_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/raw")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "mobile_item_rec_model"
REFERENCE_DATE = pd.Timestamp("2014-12-18").date()

# Feature definitions (reusable constant)
FEATURE_LIST = [
    # 1D Window - Item Stats
    "item_stats_1d:last_1d_item_clk", "item_stats_1d:last_1d_item_fav", 
    "item_stats_1d:last_1d_item_cart", "item_stats_1d:last_1d_item_buy", 
    "item_stats_1d:last_1d_item_cr", "item_stats_1d:last_1d_item_uniq_users",
    # 1D Window - User Stats
    "user_stats_1d:last_1d_user_clk", "user_stats_1d:last_1d_user_fav", 
    "user_stats_1d:last_1d_user_cart", "user_stats_1d:last_1d_user_buy", 
    "user_stats_1d:last_1d_user_cr",
    # 1D Window - User-Item Stats
    "ui_stats_1d:last_1d_ui_clk", "ui_stats_1d:last_1d_ui_fav", 
    "ui_stats_1d:last_1d_ui_cart", "ui_stats_1d:last_1d_ui_buy",
    # 1D Window - User-Category Stats
    "uc_stats_1d:last_1d_uc_clk", "uc_stats_1d:last_1d_uc_fav", 
    "uc_stats_1d:last_1d_uc_cart", "uc_stats_1d:last_1d_uc_buy",
    # 3D Window - Item Stats
    "item_stats_3d:last_3d_item_clk", "item_stats_3d:last_3d_item_fav", 
    "item_stats_3d:last_3d_item_cart", "item_stats_3d:last_3d_item_buy", 
    "item_stats_3d:last_3d_item_cr", "item_stats_3d:last_3d_item_uniq_users",
    # 3D Window - User Stats
    "user_stats_3d:last_3d_user_clk", "user_stats_3d:last_3d_user_fav", 
    "user_stats_3d:last_3d_user_cart", "user_stats_3d:last_3d_user_buy", 
    "user_stats_3d:last_3d_user_cr",
    # 3D Window - User-Item Stats
    "ui_stats_3d:last_3d_ui_clk", "ui_stats_3d:last_3d_ui_fav", 
    "ui_stats_3d:last_3d_ui_cart", "ui_stats_3d:last_3d_ui_buy",
    # 3D Window - User-Category Stats
    "uc_stats_3d:last_3d_uc_clk", "uc_stats_3d:last_3d_uc_fav", 
    "uc_stats_3d:last_3d_uc_cart", "uc_stats_3d:last_3d_uc_buy",
    # Recency
    "recency_stats:last_touch_hour", "recency_stats:days_since"
]

# Global state
ml_objects: Dict[str, Any] = {"model": None, "store": None}


def load_mlflow_model(timeout: int = 10) -> Any:
    """Load MLflow model with fallback strategy and timeout."""
    import socket
    socket.setdefaulttimeout(timeout)  # Set connection timeout
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    strategies = [
        ("alias 'champion'", f"models:/{MODEL_NAME}@champion"),
        ("stage 'Production'", f"models:/{MODEL_NAME}/Production"),
        ("latest version", None)  # Special case handled below
    ]
    
    for strategy_name, model_uri in strategies:
        try:
            if model_uri is None:  # Latest version case
                latest = client.get_latest_versions(MODEL_NAME, stages=[])[0]
                model_uri = f"models:/{MODEL_NAME}/{latest.version}"
                logger.info(f"Attempting to load model using {strategy_name}: version {latest.version}")
            else:
                logger.info(f"Attempting to load model using {strategy_name}: {model_uri}")
            
            # Try to load with sklearn first (preserves predict_proba)
            try:
                model = mlflow.sklearn.load_model(model_uri=model_uri)
                logger.info(f"✓ Model loaded successfully as sklearn model using {strategy_name}")
                return model
            except Exception as sklearn_error:
                logger.info(f"Could not load as sklearn model: {sklearn_error}, trying pyfunc...")
                # Fallback to pyfunc if sklearn doesn't work
                model = mlflow.pyfunc.load_model(model_uri=model_uri)
                logger.info(f"✓ Model loaded successfully as pyfunc model using {strategy_name}")
                return model
                
        except Exception as e:
            logger.warning(f"✗ Failed to load using {strategy_name}: {e}")
            continue
    
    raise Exception("All model loading strategies failed")


async def warmup_model():
    """Background task to warm up the model after server starts."""
    # Small delay to ensure server is fully ready for health checks
    await asyncio.sleep(2)
    
    logger.info("🔥 Starting background model warmup...")
    try:
        # Run blocking model load in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, load_mlflow_model, 120)
        ml_objects["model"] = model
        logger.info("✓ Model warmed up successfully in background")
    except Exception as e:
        logger.warning(f"⚠ Background warmup failed: {e}")
        logger.warning("Model will be loaded on first prediction request instead")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting API Lifespan")

    # Load Feature Store (local, fast)
    try:
        logger.info(f"Connecting to Feature Store at {FEATURE_REPO_PATH}...")
        ml_objects["store"] = FeatureStore(repo_path=FEATURE_REPO_PATH)
        logger.info("✓ Feature Store connected successfully")
    except Exception as e:
        logger.error(f"✗ CRITICAL: Failed to load Feature Store: {e}")

    # Start background warmup task (non-blocking)
    # This allows the server to start immediately and pass health checks
    # while the model loads in the background
    warmup_task = asyncio.create_task(warmup_model())

    yield

    # Cancel warmup if still running during shutdown
    warmup_task.cancel()
    logger.info("🛑 API Lifespan ended")
    ml_objects.clear()


# FastAPI app initialization
app = FastAPI(
    title="Mobile Item Recommendation Service",
    description="Real-time inference using Feast + XGBoost",
    lifespan=lifespan
)
Instrumentator().instrument(app).expose(app)


# Request models
class RecommendationRequest(BaseModel):
    user_id: int
    item_id: int
    item_category: int


class UserRecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10
    min_probability: float = 0.0


def prepare_features(feature_vector: dict) -> pd.DataFrame:
    """Prepare features for model prediction with correct data types and column order."""
    df = pd.DataFrame.from_dict(feature_vector)
    
    # Drop entity columns
    entity_cols = ['user_id', 'item_id', 'item_category', 'user_item_id', 'user_category_id']
    df = df.drop(columns=entity_cols, errors='ignore')
    
    # Handle missing values
    df = df.fillna(0)
    
    # Define expected data types based on model schema
    # Conversion rate (cr) features are doubles, everything else is long/int
    double_features = ['last_1d_item_cr', 'last_1d_user_cr', 'last_3d_item_cr', 'last_3d_user_cr']
    int32_features = ['last_touch_hour']
    int64_features = ['days_since']
    
    # Convert all columns to appropriate types
    for col in df.columns:
        if col in double_features:
            # Keep as float64 (double)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
        elif col in int32_features:
            # Convert to int32
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
        elif col in int64_features:
            # Convert to int64
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
        else:
            # All other count-based features should be int64 (long)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    
    # Define the expected column order (matching training data)
    expected_columns = [
        'last_1d_item_clk', 'last_1d_item_fav', 'last_1d_item_cart', 'last_1d_item_buy',
        'last_1d_item_cr', 'last_1d_item_uniq_users', 'last_1d_user_clk', 'last_1d_user_fav',
        'last_1d_user_cart', 'last_1d_user_buy', 'last_1d_user_cr', 'last_1d_ui_clk',
        'last_1d_ui_fav', 'last_1d_ui_cart', 'last_1d_ui_buy', 'last_1d_uc_clk',
        'last_1d_uc_fav', 'last_1d_uc_cart', 'last_1d_uc_buy', 'last_3d_item_clk',
        'last_3d_item_fav', 'last_3d_item_cart', 'last_3d_item_buy', 'last_3d_item_cr',
        'last_3d_item_uniq_users', 'last_3d_user_clk', 'last_3d_user_fav', 'last_3d_user_cart',
        'last_3d_user_buy', 'last_3d_user_cr', 'last_3d_ui_clk', 'last_3d_ui_fav',
        'last_3d_ui_cart', 'last_3d_ui_buy', 'last_3d_uc_clk', 'last_3d_uc_fav',
        'last_3d_uc_cart', 'last_3d_uc_buy', 'last_touch_hour', 'days_since'
    ]
    
    # Reorder columns to match training data
    # Add missing columns with 0 values if any
    for col in expected_columns:
        if col not in df.columns:
            if col in double_features:
                df[col] = 0.0
            elif col in int32_features:
                df[col] = pd.array([0] * len(df), dtype='int32')
            else:
                df[col] = pd.array([0] * len(df), dtype='int64')
    
    # Select and reorder columns
    df = df[expected_columns]
    
    return df


def get_candidate_items(user_id: int, days_back: int = 1) -> pd.DataFrame:
    """Get candidate items for a user based on recent interactions."""
    raw_users_path = f"{DATA_PATH}/users.parquet"
    
    if not os.path.exists(raw_users_path):
        raise FileNotFoundError(f"Raw data not found at {raw_users_path}")
    
    df = pd.read_parquet(raw_users_path)
    df['date'] = pd.to_datetime(df['time']).dt.date
    
    start_date = REFERENCE_DATE - timedelta(days=days_back)
    
    logger.info(f"Searching for user {user_id} interactions: {start_date} to {REFERENCE_DATE}")
    
    # Filter user interactions
    user_interactions = df[
        (df['user_id'] == user_id) &
        (df['date'] >= start_date) &
        (df['date'] < REFERENCE_DATE)
    ]
    
    if len(user_interactions) == 0 and days_back == 1:
        # Fallback to 7 days
        logger.info(f"No interactions in last day, trying last 7 days for user {user_id}")
        return get_candidate_items(user_id, days_back=7)
    
    candidates = user_interactions[['user_id', 'item_id', 'item_category']].drop_duplicates()
    logger.info(f"Found {len(candidates)} candidate items for user {user_id}")
    
    return candidates


def validate_service() -> tuple[Any, Any]:
    """Validate that model and store are loaded, with lazy loading."""
    store = ml_objects["store"]
    
    if not store:
        raise HTTPException(status_code=503, detail="Feature Store is not connected")
    
    # Lazy load model if not yet loaded
    if not ml_objects["model"]:
        logger.info("Model not loaded, attempting lazy load...")
        try:
            # Use longer timeout (60s) for first load - model download can be slow
            ml_objects["model"] = load_mlflow_model(timeout=60)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")
    
    return ml_objects["model"], store


@app.post("/predict", tags=["Inference"])
def predict(request: RecommendationRequest):
    """Predict buy probability for a single user-item pair."""
    model, store = validate_service()

    try:
        # Create composite keys
        user_item_id = f"{request.user_id}_{request.item_id}"
        user_category_id = f"{request.user_id}_{request.item_category}"
        
        # Fetch features
        feature_vector = store.get_online_features(
            features=FEATURE_LIST,
            entity_rows=[{
                "user_id": request.user_id,
                "item_id": request.item_id,
                "user_item_id": user_item_id,
                "user_category_id": user_category_id
            }]
        ).to_dict()

        # Prepare and predict
        input_df = prepare_features(feature_vector)
        prediction = model.predict(input_df)
        prob = float(model.predict_proba(input_df)[0][1])
        
        logger.info(f"Prediction for user {request.user_id}, item {request.item_id}: {prediction[0]}, prob: {prob:.4f}")
        
        return {
            "status": "success",
            "user_id": request.user_id,
            "item_id": request.item_id,
            "prediction": int(prediction[0]),
            "buy_probability": prob
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", tags=["Inference"])
def recommend(request: UserRecommendationRequest):
    """Get top-k item recommendations for a user."""
    model, store = validate_service()

    try:
        # Get candidate items
        candidates = get_candidate_items(request.user_id)
        
        if len(candidates) == 0:
            return {
                "status": "success",
                "user_id": request.user_id,
                "recommendations": [],
                "total_candidates": 0,
                "message": "No candidate items found for this user"
            }

        # Prepare entity rows
        entity_rows = [
            {
                "user_id": int(row['user_id']),
                "item_id": int(row['item_id']),
                "user_item_id": f"{row['user_id']}_{row['item_id']}",
                "user_category_id": f"{row['user_id']}_{row['item_category']}"
            }
            for _, row in candidates.iterrows()
        ]

        # Batch fetch features
        logger.info(f"Fetching features for {len(entity_rows)} candidates...")
        feature_vector = store.get_online_features(
            features=FEATURE_LIST,
            entity_rows=entity_rows
        ).to_dict()

        # Prepare and predict
        input_df = prepare_features(feature_vector)
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]

        # Create results
        results_df = pd.DataFrame({
            'item_id': candidates['item_id'].values[:len(predictions)],
            'item_category': candidates['item_category'].values[:len(predictions)],
            'prediction': predictions,
            'buy_probability': probabilities
        })

        # Filter and sort
        results_df = results_df[results_df['buy_probability'] >= request.min_probability]
        results_df = results_df.sort_values('buy_probability', ascending=False).head(request.top_k)

        # Format response
        recommendations = [
            {
                "item_id": int(row['item_id']),
                "item_category": int(row['item_category']),
                "buy_probability": float(row['buy_probability'])
            }
            for _, row in results_df.iterrows()
        ]

        logger.info(f"Returning {len(recommendations)} recommendations for user {request.user_id}")

        return {
            "status": "success",
            "user_id": request.user_id,
            "recommendations": recommendations,
            "total_candidates": len(candidates),
            "filtered_results": len(results_df)
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint - returns OK for startup probe."""
    return "OK"


@app.get("/health/detailed", tags=["Health"])
def health_detailed():
    """Detailed health check endpoint."""
    model_loaded = ml_objects["model"] is not None
    store_connected = ml_objects["store"] is not None
    
    return {
        "status": "healthy" if (model_loaded and store_connected) else "degraded",
        "model_loaded": model_loaded,
        "feature_store_connected": store_connected
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Get detailed model information."""
    if not ml_objects["model"]:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    model = ml_objects["model"]
    info = {"model_type": type(model).__name__}
    
    try:
        if hasattr(model, "metadata"):
            info["metadata"] = str(model.metadata)
        
        if hasattr(model, "model"):
            underlying = model.model
            info["underlying_model_type"] = underlying.__class__.__name__
            
            if hasattr(underlying, "get_booster"):
                booster = underlying.get_booster()
                if hasattr(booster, "num_boosted_rounds"):
                    info["num_trees"] = booster.num_boosted_rounds()
            
            if hasattr(underlying, "n_estimators"):
                info["n_estimators"] = underlying.n_estimators
            if hasattr(underlying, "max_depth"):
                info["max_depth"] = underlying.max_depth
    except Exception as e:
        info["error"] = str(e)
    
    return info