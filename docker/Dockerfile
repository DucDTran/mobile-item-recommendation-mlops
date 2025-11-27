FROM python:3.11-slim

# libpq-dev is for Postgres, build-essential for compiling C extensions (e.g. Pandas/Feast)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/app/airflow \
    FEAST_USAGE=False

WORKDIR /app

COPY requirements.txt .

# Install remaining dependencies with Airflow constraints
RUN pip install --no-cache-dir -r requirements.txt --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"

COPY src/ /app/src/
COPY feature_repo/ /app/feature_repo/
COPY airflow/ /app/airflow/
COPY docker/init-mlflow-db.sh /app/init-mlflow-db.sh
RUN chmod +x /app/init-mlflow-db.sh
# CMD ["bash"]
ENTRYPOINT ["/app/init-mlflow-db.sh"]
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]