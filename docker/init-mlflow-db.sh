#!/bin/bash
set -e

POSTGRES_HOST=${POSTGRES_HOST:-postgres}

echo "Waiting for PostgreSQL to be ready..."
for i in $(seq 1 60); do
  if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1" >/dev/null 2>&1; then
    break
  fi
  echo "Postgres not ready yet... retrying"
  sleep 2
done

# Optional full reset (use carefully!)
if [ "${RESET_MLFLOW_DB:-false}" = "true" ]; then
  echo "Dropping and recreating database $POSTGRES_DB ..."
  PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d postgres -c "DROP DATABASE IF EXISTS $POSTGRES_DB;" || true
  PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE $POSTGRES_DB;" || true
fi

# Cleanup stale Alembic revisions if present
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "DELETE FROM alembic_version WHERE version_num='686269002441';" >/dev/null 2>&1 || true

echo "Starting MLflow server..."
exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000