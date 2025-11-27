#!/bin/bash
set -e

# Create additional databases
# The primary database (POSTGRES_DB) is created automatically by the postgres image
# This script creates any additional databases needed

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE airflow_db;
    GRANT ALL PRIVILEGES ON DATABASE airflow_db TO "$POSTGRES_USER";
EOSQL

echo "Additional databases created successfully"
