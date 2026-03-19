#!/bin/bash
# Create the 'hololoom' database for Memory Bus tables.
# The default 'mem0' database is created by POSTGRES_DB env var.
# This script runs as part of docker-entrypoint-initdb.d on first boot.

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE hololoom OWNER hololoom'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'hololoom')\gexec
EOSQL

echo "Database 'hololoom' created (or already exists)."
