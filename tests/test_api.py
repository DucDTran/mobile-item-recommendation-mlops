import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from src.app import app, ml_objects

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check_unloaded(client):
    ml_objects["model"] = None
    ml_objects["store"] = None
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False
    assert response.json()["feature_store_connected"] is False

def test_predict_success(client):
    
    mock_store = MagicMock()

    mock_store.get_online_features.return_value.to_dict.return_value = {
        "last_1d_item_clk": [5.0],
        "last_1d_item_fav": [1.0],
        "last_1d_item_cart": [1.0],
        "last_1d_item_buy": [1.0],
        "last_1d_user_clk": [10.0],
        "last_1d_user_fav": [1.0],
        "last_1d_user_cart": [1.0],
        "last_1d_user_buy": [1.0],
        "last_3d_item_clk": [15.0],
        "last_3d_item_fav": [2.0],
        "last_3d_item_cart": [2.0],
        "last_3d_item_buy": [2.0],
        "last_3d_user_clk": [20.0],
        "last_3d_user_fav": [3.0],
        "last_3d_user_cart": [3.0],
        "last_3d_user_buy": [3.0],
        "last_3d_ui_clk": [30.0],
        "last_3d_ui_fav": [4.0],
        "last_3d_ui_cart": [4.0],
        "last_3d_ui_buy": [4.0],
        "last_3d_uc_clk": [40.0],
        "last_3d_uc_fav": [5.0],
        "last_3d_uc_cart": [5.0],
        "last_3d_uc_buy": [5.0],
        "last_3d_item_uniq_users": [6.0],
        "last_3d_user_uniq_items": [7.0],
        "last_3d_ui_uniq_items": [8.0],
        "last_3d_uc_uniq_items": [9.0],
        "last_3d_item_cr": [0.2],
        "last_3d_user_cr": [0.3],
        "last_3d_ui_cr": [0.4],
        "last_3d_uc_cr": [0.5],
        "last_3d_item_uniq_users": [6.0],
        "last_3d_user_uniq_items": [7.0],
        "last_3d_ui_uniq_items": [8.0],
        "last_3d_uc_uniq_items": [9.0],
        "last_3d_item_cr": [0.2],
        "last_3d_user_cr": [0.3],
        "last_3d_ui_cr": [0.4],
        "last_3d_uc_cr": [0.5],
        "last_touch_hour": [14],
        "days_since": [2]
    }
    
    # 2. Mock the MLflow Model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    ml_objects["store"] = mock_store
    ml_objects["model"] = mock_model

    payload = {
        "user_id": 100,
        "item_id": 200,
        "item_category": 5
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"] == 1
    assert data["buy_probability"] == 0.8
    
    mock_store.get_online_features.assert_called_once()

def test_predict_missing_model(client):
    ml_objects["model"] = None
    ml_objects["store"] = MagicMock()
    
    payload = {"user_id": 1, "item_id": 2, "item_category": 3}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]