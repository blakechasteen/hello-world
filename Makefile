.PHONY: help setup up down restart logs shell test clean

# Default target
help:
	@echo "LMS Orchestration - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Initial setup (copy .env, build images)"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo ""
	@echo "Development:"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View logs (all services)"
	@echo "  make shell        - Open shell in API container"
	@echo "  make db-shell     - Open PostgreSQL shell"
	@echo "  make neo4j-shell  - Open Neo4j shell"
	@echo ""
	@echo "Database:"
	@echo "  make migrate      - Run database migrations"
	@echo "  make seed         - Seed initial data"
	@echo "  make db-backup    - Backup all databases"
	@echo "  make db-restore   - Restore from backups"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests"
	@echo "  make test-int     - Run integration tests"
	@echo "  make test-e2e     - Run end-to-end tests"
	@echo "  make lint         - Run linters"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove containers and volumes"
	@echo "  make reset        - Complete reset (DANGER: deletes all data)"

# Setup
setup:
	@echo "Setting up LMS development environment..."
	cp -n .env.example .env || true
	docker-compose build
	@echo "Setup complete! Run 'make up' to start services."

# Start services
up:
	docker-compose up -d
	@echo "Services started. Run 'make logs' to view logs."

# Stop services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-postgres:
	docker-compose logs -f postgres

logs-neo4j:
	docker-compose logs -f neo4j

# Shells
shell:
	docker-compose exec api bash

db-shell:
	docker-compose exec postgres psql -U lms_user -d lms_dev

neo4j-shell:
	docker-compose exec neo4j cypher-shell -u neo4j -p lms_dev_password

redis-shell:
	docker-compose exec redis redis-cli

# Database operations
migrate:
	docker-compose exec api alembic upgrade head

migrate-down:
	docker-compose exec api alembic downgrade -1

migrate-create:
	@read -p "Migration name: " name; \
	docker-compose exec api alembic revision --autogenerate -m "$$name"

seed:
	docker-compose exec api python scripts/seed_data.py

db-backup:
	@mkdir -p backups
	@echo "Backing up PostgreSQL..."
	docker-compose exec -T postgres pg_dump -U lms_user lms_dev > backups/postgres_$(date +%Y%m%d_%H%M%S).sql
	@echo "Backing up Neo4j..."
	docker-compose exec neo4j neo4j-admin dump --to=/tmp/backup.dump
	docker cp lms-neo4j:/tmp/backup.dump backups/neo4j_$(date +%Y%m%d_%H%M%S).dump
	@echo "Backups complete in ./backups/"

db-restore:
	@echo "Not implemented yet. Use docker cp and restore commands."

# Testing
test:
	docker-compose exec api pytest tests/ -v

test-unit:
	docker-compose exec api pytest tests/unit/ -v

test-int:
	docker-compose exec api pytest tests/integration/ -v

test-e2e:
	docker-compose exec api pytest tests/e2e/ -v

test-cov:
	docker-compose exec api pytest tests/ --cov=lms_api --cov-report=html

lint:
	docker-compose exec api ruff check .
	docker-compose exec api mypy .

format:
	docker-compose exec api ruff format .

# Cleanup
clean:
	docker-compose down -v
	rm -rf logs/*.log

reset:
	@echo "WARNING: This will delete ALL data!"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		docker-compose down -v; \
		rm -rf logs/*.log backups/*; \
		echo "Reset complete."; \
	else \
		echo "Cancelled."; \
	fi

# Profiles
up-dev-tools:
	docker-compose --profile dev-tools up -d

up-monitoring:
	docker-compose --profile monitoring up -d

up-llm:
	docker-compose --profile llm up -d

# Status
ps:
	docker-compose ps

stats:
	docker stats --no-stream

health:
	@echo "Checking service health..."
	@docker-compose exec postgres pg_isready -U lms_user && echo "✓ PostgreSQL is ready" || echo "✗ PostgreSQL is not ready"
	@docker-compose exec redis redis-cli ping | grep -q PONG && echo "✓ Redis is ready" || echo "✗ Redis is not ready"
	@docker-compose exec neo4j cypher-shell -u neo4j -p lms_dev_password "RETURN 1" > /dev/null 2>&1 && echo "✓ Neo4j is ready" || echo "✗ Neo4j is not ready"
	@curl -s http://localhost:6333/health | grep -q "ok" && echo "✓ Qdrant is ready" || echo "✗ Qdrant is not ready"
