#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
sleep 5

# Check if database needs migration reset
if ! mlflow db upgrade $MLFLOW_BACKEND_STORE_URI 2>/dev/null; then
    echo "Database migration failed. Attempting to reset..."
    
    # Try to reset alembic version
    PGPASSWORD=$POSTGRES_PASSWORD psql -h postgres -U $POSTGRES_USER -d $POSTGRES_DB -c "DELETE FROM alembic_version;" 2>/dev/null || true
    
    # Try upgrade again
    mlflow db upgrade $MLFLOW_BACKEND_STORE_URI || true
fi

echo "Starting MLflow server..."
exec mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000