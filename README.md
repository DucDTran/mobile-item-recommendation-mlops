# 🛒 Mobile Item Recommendation System

A production-ready **MLOps pipeline** for real-time item purchase prediction on mobile e-commerce platforms. Built with modern ML infrastructure including feature stores, experiment tracking, orchestration, and observability.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/MLflow-2.3+-orange?logo=mlflow" alt="MLflow">
  <img src="https://img.shields.io/badge/Feast-0.30+-purple" alt="Feast">
  <img src="https://img.shields.io/badge/Airflow-2.9.3-red?logo=apache-airflow" alt="Airflow">
  <img src="https://img.shields.io/badge/XGBoost-ML-yellow" alt="XGBoost">
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?logo=docker" alt="Docker">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Feature Engineering](#-feature-engineering)
- [Model Training](#-model-training)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [Testing](#-testing)
- [Development](#-development)

---

## 🎯 Overview

This project implements an end-to-end MLOps system that predicts whether a user will purchase a specific item on a mobile e-commerce platform. The system uses historical user behavior data from the **Tianchi Fresh Competition** dataset.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Real-time Inference** | Sub-100ms predictions via FastAPI + Redis feature store |
| **Feature Store** | Feast-managed features with offline/online sync |
| **Experiment Tracking** | MLflow for model versioning, metrics, and artifacts |
| **Pipeline Orchestration** | Airflow DAGs for automated retraining |
| **Auto-Tuning** | Optuna hyperparameter optimization |
| **Observability** | Prometheus + Grafana metrics dashboards |
| **CI/CD** | GitHub Actions → GCP Cloud Run + VM deployment |

### Problem Statement

Given a user's historical interactions (clicks, favorites, cart additions, purchases), predict the probability that they will buy a specific item on a given day.

**Target Variable:** Binary classification (1 = purchase, 0 = no purchase)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MLOps Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│   │   Raw Data   │────▶│ Preprocessing│────▶│   Features   │                    │
│   │  (Parquet)   │     │   (Python)   │     │  (Parquet)   │                    │
│   └──────────────┘     └──────────────┘     └──────────────┘                    │
│                                                    │                             │
│                        ┌───────────────────────────┼───────────────────────────┐ │
│                        │                           ▼                           │ │
│   ┌──────────────┐     │  ┌──────────────┐   ┌──────────────┐                  │ │
│   │   Airflow    │────▶│  │    Feast     │──▶│    Redis     │                  │ │
│   │ (Scheduler)  │     │  │  (Registry)  │   │(Online Store)│                  │ │
│   └──────────────┘     │  └──────────────┘   └──────────────┘                  │ │
│         │              │                           │                           │ │
│         │              │        Feature Store      │                           │ │
│         │              └───────────────────────────┼───────────────────────────┘ │
│         │                                          │                             │
│         ▼                                          ▼                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│   │   Training   │────▶│    MLflow    │────▶│   FastAPI    │◀───── Requests    │
│   │  (XGBoost)   │     │  (Registry)  │     │   (Server)   │                    │
│   └──────────────┘     └──────────────┘     └──────────────┘                    │
│                              │                     │                             │
│                              │                     ▼                             │
│                        ┌─────┴─────┐        ┌──────────────┐                    │
│                        │ PostgreSQL│        │  Prometheus  │──▶ Grafana         │
│                        └───────────┘        └──────────────┘                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Ingestion**: Raw user behavior data stored as Parquet files
2. **Feature Engineering**: Time-windowed aggregations (1-day, 3-day windows)
3. **Feature Materialization**: Feast syncs features to Redis for low-latency serving
4. **Model Training**: XGBoost with Optuna tuning, logged to MLflow
5. **Model Serving**: FastAPI loads champion model from MLflow registry
6. **Inference**: Real-time feature retrieval from Feast → Model prediction

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **ML Framework** | XGBoost, scikit-learn | Model training and baselines |
| **Feature Store** | Feast + Redis | Feature management and online serving |
| **Experiment Tracking** | MLflow | Model versioning, metrics, artifacts |
| **Orchestration** | Apache Airflow | Pipeline scheduling and automation |
| **Hyperparameter Tuning** | Optuna | Automated hyperparameter optimization |
| **API Framework** | FastAPI | High-performance REST API |
| **Monitoring** | Prometheus + Grafana | Metrics collection and visualization |
| **Database** | PostgreSQL | MLflow/Airflow backend storage |
| **Containerization** | Docker + Docker Compose | Service orchestration |
| **Cloud** | GCP (Cloud Run, Compute Engine) | Production deployment |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

---

## 📁 Project Structure

```
mobile-item-recommendation/
├── 📂 airflow/
│   └── dags/
│       └── pipeline.py          # Airflow DAG: preprocess → materialize → train
│
├── 📂 data/
│   ├── raw/                     # Raw data files (Parquet)
│   │   ├── users.parquet        # User interaction logs
│   │   └── items.parquet        # Item metadata
│   └── features/                # Generated feature tables
│       ├── item_stats_*.parquet
│       ├── user_stats_*.parquet
│       ├── ui_stats_*.parquet   # User-Item interactions
│       ├── uc_stats_*.parquet   # User-Category interactions
│       └── recency_stats_*.parquet
│
├── 📂 docker/
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── Dockerfile               # Python 3.11 base image
│   ├── prometheus.yml           # Prometheus scrape config
│   └── init-*.sh                # Database initialization scripts
│
├── 📂 feature_repo/
│   ├── feature_store.yaml       # Feast configuration
│   ├── feature_definitions.py   # Feature views and entities
│   └── data/
│       └── registry.db          # Feast registry
│
├── 📂 notebooks/
│   └── experiment.ipynb         # Data exploration and analysis
│
├── 📂 src/
│   ├── app.py                   # FastAPI application
│   ├── preprocessing.py         # Feature generation pipeline
│   └── train.py                 # Model training with MLflow
│
├── 📂 tests/
│   ├── test_api.py              # API endpoint tests
│   └── test_preprocessing.py    # Feature generation tests
│
├── 📂 .github/workflows/
│   └── deploy.yml               # CI/CD pipeline
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mobile-item-recommendation.git
cd mobile-item-recommendation
```

### 2. Start All Services

```bash
cd docker
docker-compose up -d
```

This starts:
| Service | Port | Description |
|---------|------|-------------|
| FastAPI | 8000 | Model serving API |
| MLflow | 5000 | Experiment tracking UI |
| Airflow | 8080 | Pipeline orchestration UI |
| Grafana | 3000 | Metrics dashboards |
| Prometheus | 9090 | Metrics collection |
| PostgreSQL | 5432 | Backend database |
| Redis | 6379 | Feature store cache |

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | - |
| **MLflow UI** | http://localhost:5000 | - |
| **Airflow UI** | http://localhost:8080 | admin / admin |
| **Grafana** | http://localhost:3000 | admin / admin |

### 4. Run the Pipeline

```bash
# Option 1: Trigger via Airflow UI
# Navigate to http://localhost:8080 → Enable "mobile_item_rec_pipeline" → Trigger

# Option 2: Run components manually
docker exec -it mlops_airflow_scheduler bash
python /app/src/preprocessing.py
cd /app/feature_repo && feast apply && feast materialize 2014-11-01T00:00:00 2015-01-01T00:00:00
python /app/src/train.py
```

### 5. Make Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 100, "item_id": 200, "item_category": 5}'

# Get recommendations for a user
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 100, "top_k": 10, "min_probability": 0.5}'
```

---

## 📡 API Reference

### Endpoints

#### `POST /predict`
Predict purchase probability for a single user-item pair.

**Request:**
```json
{
  "user_id": 100,
  "item_id": 200,
  "item_category": 5
}
```

**Response:**
```json
{
  "status": "success",
  "user_id": 100,
  "item_id": 200,
  "prediction": 1,
  "buy_probability": 0.847
}
```

#### `POST /recommend`
Get top-k item recommendations for a user.

**Request:**
```json
{
  "user_id": 100,
  "top_k": 10,
  "min_probability": 0.5
}
```

**Response:**
```json
{
  "status": "success",
  "user_id": 100,
  "recommendations": [
    {"item_id": 200, "item_category": 5, "buy_probability": 0.92},
    {"item_id": 350, "item_category": 12, "buy_probability": 0.87}
  ],
  "total_candidates": 45,
  "filtered_results": 10
}
```

#### `GET /health`
Basic health check for load balancers.

**Response:** `"OK"` (status 200)

#### `GET /health/detailed`
Detailed health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_store_connected": true
}
```

#### `GET /model/info`
Get information about the loaded model.

**Response:**
```json
{
  "model_type": "XGBClassifier",
  "n_estimators": 150,
  "max_depth": 6
}
```

---

## 🔧 Feature Engineering

### Feature Categories

The system generates **40 features** across multiple time windows and entity types:

| Category | Window | Features | Description |
|----------|--------|----------|-------------|
| **Item Stats** | 1d, 3d | clicks, favs, carts, buys, CR, unique_users | Item popularity metrics |
| **User Stats** | 1d, 3d | clicks, favs, carts, buys, CR | User activity metrics |
| **User-Item Stats** | 1d, 3d | clicks, favs, carts, buys | Specific user-item interactions |
| **User-Category Stats** | 1d, 3d | clicks, favs, carts, buys | User's category preferences |
| **Recency** | - | last_touch_hour, days_since | Time since last interaction |

### Feature Generation

Features are computed using time-windowed aggregations:

```python
# Example: 1-day item statistics
item_stats = df.groupby(['item_id', 'behavior_type']).size().unstack(fill_value=0)
item_stats.columns = ['clicks', 'favs', 'carts', 'buys']
item_stats['conversion_rate'] = buys / (clicks + 1)
```

### Feast Feature Views

Features are registered in Feast with the following entities:

```yaml
Entities:
  - user (join_key: user_id)
  - item (join_key: item_id)
  - user_item (join_key: user_item_id)      # Composite: "{user_id}_{item_id}"
  - user_category (join_key: user_category_id)  # Composite: "{user_id}_{category}"
```

---

## 🎓 Model Training

### Training Pipeline

The training pipeline (`src/train.py`) executes:

1. **Dataset Construction**: Build train/val/test sets from feature tables
2. **Downsampling**: Balance class distribution (20:1 ratio)
3. **Baseline Models**: Train Logistic Regression, Random Forest, XGBoost
4. **Hyperparameter Tuning**: Optuna optimization (30 trials)
5. **Evaluation**: Local and Global precision/recall/F1
6. **Model Registration**: Best model promoted to MLflow "champion"

### Model Selection

Models are compared using **Local F1** on the validation set:

```python
# Optuna search space
params = {
    "n_estimators": (50, 300),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "scale_pos_weight": (1, 10)
}
```

### MLflow Integration

```python
# Training logs to MLflow
mlflow.set_experiment("mobile-item-recommendation")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({"val_f1": f1_score, "val_precision": precision})
    mlflow.sklearn.log_model(model, "model", signature=signature)
    
# Best model gets "champion" alias
client.set_registered_model_alias("mobile_item_rec_model", "champion", version)
```

### Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Local Precision** | TP / Predicted Positives (in candidate set) |
| **Local Recall** | TP / Actual Positives (in candidate set) |
| **Global Recall** | TP / All Purchases (that day) |
| **F1 Score** | Harmonic mean of Precision and Recall |

---

## 🚢 Deployment

### Local Development

```bash
# Start all services
cd docker && docker-compose up -d

# View logs
docker-compose logs -f fastapi
```

### Production (GCP)

The CI/CD pipeline (`.github/workflows/deploy.yml`) deploys to:

1. **GCP Compute Engine VM**: Full stack (MLflow, Airflow, Redis, Postgres)
2. **GCP Cloud Run**: Stateless API (auto-scaling, serverless)

#### Deployment Flow

```
Push to main
     │
     ▼
┌─────────────────┐
│  Build & Push   │──▶ Docker image → GCP Artifact Registry
│  Docker Image   │
└─────────────────┘
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
┌─────────────────┐            ┌─────────────────┐
│  Deploy to VM   │            │ Deploy to Cloud │
│  (Full Stack)   │            │      Run        │
└─────────────────┘            └─────────────────┘
     │                                  │
     ▼                                  ▼
  MLflow, Airflow,              FastAPI API
  Redis, Prometheus             (auto-scaled)
```

#### Required Secrets

Configure in GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT` | GCP project ID |
| `GCP_CREDENTIALS` | Service account JSON key |
| `GCP_VM_IP` | Compute Engine VM IP |
| `SSH_USERNAME` | VM SSH username |
| `SSH_PRIVATE_KEY` | SSH private key for VM access |

---

## 📊 Monitoring

### Prometheus Metrics

FastAPI exposes metrics at `/metrics`:

- `http_requests_total`: Request count by endpoint/status
- `http_request_duration_seconds`: Request latency histogram
- `http_request_size_bytes`: Request payload sizes
- `http_response_size_bytes`: Response payload sizes

### Grafana Dashboards

Access Grafana at `http://localhost:3000`:

1. Add Prometheus data source: `http://prometheus:9090`
2. Import dashboards for:
   - API latency (p50, p95, p99)
   - Request throughput
   - Error rates
   - Model prediction distributions

### Health Checks

```bash
# Basic health (for load balancers)
curl http://localhost:8000/health

# Detailed health
curl http://localhost:8000/health/detailed
```

---

## 🧪 Testing

### Run Tests

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

| File | Coverage |
|------|----------|
| `test_api.py` | FastAPI endpoints, model mocking |
| `test_preprocessing.py` | Feature generation logic |

### Example Test

```python
def test_predict_success(client):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    ml_objects["model"] = mock_model
    
    response = client.post("/predict", json={"user_id": 100, "item_id": 200, "item_category": 5})
    assert response.json()["buy_probability"] == 0.8
```

---

## 💻 Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI locally
uvicorn src.app:app --reload --port 8000
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `FEAST_REPO_PATH` | `/app/feature_repo` | Feast repository path |
| `DATA_PATH` | `/app/data/raw` | Raw data directory |
| `FEATURE_PATH` | `/app/data/features` | Feature output directory |

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

---

## 📚 Dataset

This project uses the **Tianchi Fresh Competition** dataset containing mobile e-commerce user behavior data.

### Data Schema

**User Interactions (`users.parquet`)**

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Unique user identifier |
| `item_id` | int | Item identifier |
| `item_category` | int | Item category |
| `behavior_type` | int | 1=click, 2=favorite, 3=cart, 4=buy |
| `time` | datetime | Interaction timestamp |

### Date Splits

| Split | Date | Purpose |
|-------|------|---------|
| Train | 2014-12-06 | Model training |
| Validation | 2014-12-12 | Hyperparameter tuning |
| Test | 2014-12-18 | Final evaluation |

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Tianchi platform for providing the dataset
- MLflow, Feast, and Airflow communities
- FastAPI and XGBoost maintainers

---

<p align="center">
  Built with ❤️ for MLOps best practices
</p>
