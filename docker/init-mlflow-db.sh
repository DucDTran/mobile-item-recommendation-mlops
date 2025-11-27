#!/bin/bash
set -e

POSTGRES_HOST=${POSTGRES_HOST:-postgres}

echo "Waiting for PostgreSQL to be ready..."
until PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1" >/dev/null 2>&1; do
  sleep 2
done

echo "Applying MLflow migrations..."
if ! mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI"; then
  echo "Database migration failed. Attempting to reset alembic_version..."
  PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "DELETE FROM alembic_version;" || true
  mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI"
fi

echo "Starting MLflow server..."
exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000