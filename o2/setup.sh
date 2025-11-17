#!/bin/bash
set -e

# O2 Platform Setup Script
# Platform Anarchism meets Agentic Intelligence

echo "========================================="
echo "O2 Platform Setup"
echo "Platform Anarchism + HoloLoom + Matrix"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"
echo -e "${GREEN}✓ Docker Compose found${NC}"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env

    # Generate secure passwords
    POSTGRES_PW=$(openssl rand -base64 32)
    O2_DB_PW=$(openssl rand -base64 32)
    O2_BOT_PW=$(openssl rand -base64 32)
    NEO4J_PW=$(openssl rand -base64 32)

    # Update .env with generated passwords
    sed -i "s/changeme_postgres_password/$POSTGRES_PW/g" .env
    sed -i "s/changeme_o2_db_password/$O2_DB_PW/g" .env
    sed -i "s/changeme_generate_secure_password/$O2_BOT_PW/g" .env
    sed -i "s/changeme_neo4j_password/$NEO4J_PW/g" .env

    echo -e "${GREEN}✓ Created .env with secure passwords${NC}"
    echo ""
else
    echo -e "${YELLOW}! .env already exists, skipping creation${NC}"
    echo ""
fi

# Ask deployment mode
echo "Select deployment mode:"
echo "1) Development (local testing, no SSL)"
echo "2) Production (public instance, SSL, Neo4j, Qdrant)"
read -p "Enter choice [1-2]: " DEPLOY_MODE

if [ "$DEPLOY_MODE" = "2" ]; then
    PROFILE="production"
    echo ""
    read -p "Enter your domain name (e.g., matrix.example.com): " DOMAIN_NAME
    read -p "Enter your email for Let's Encrypt: " LETSENCRYPT_EMAIL

    # Update .env with production settings
    echo "DOMAIN_NAME=$DOMAIN_NAME" >> .env
    echo "LETSENCRYPT_EMAIL=$LETSENCRYPT_EMAIL" >> .env

    sed -i "s/MATRIX_SERVER_NAME=matrix.localhost/MATRIX_SERVER_NAME=$DOMAIN_NAME/g" .env
    sed -i "s/HOLOLOOM_MEMORY_BACKEND=inmemory/HOLOLOOM_MEMORY_BACKEND=hybrid/g" .env

    echo -e "${GREEN}✓ Production mode configured${NC}"
else
    PROFILE=""
    echo -e "${GREEN}✓ Development mode configured${NC}"
fi
echo ""

# Create required directories
echo "Creating data directories..."
mkdir -p synapse-data synapse-config postgres-data redis-data o2-data o2-config user-memories
mkdir -p neo4j-data neo4j-logs qdrant-data nginx-config certbot-data certbot-certs

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Generate Synapse config if it doesn't exist
if [ ! -f synapse-config/homeserver.yaml ]; then
    echo "Generating Synapse configuration..."

    # Load MATRIX_SERVER_NAME from .env
    source .env

    docker run -it --rm \
        -v "$(pwd)/synapse-data:/data" \
        -e SYNAPSE_SERVER_NAME="$MATRIX_SERVER_NAME" \
        -e SYNAPSE_REPORT_STATS=no \
        matrixdotorg/synapse:latest generate

    # Move config to synapse-config directory
    mv synapse-data/homeserver.yaml synapse-config/
    mv synapse-data/*.log.config synapse-config/ 2>/dev/null || true

    echo -e "${GREEN}✓ Synapse configuration generated${NC}"
else
    echo -e "${YELLOW}! Synapse config already exists, skipping generation${NC}"
fi
echo ""

# Create PostgreSQL init script
if [ ! -f init-db.sql ]; then
    echo "Creating PostgreSQL init script..."
    cat > init-db.sql << 'EOF'
-- Create O2 database
CREATE DATABASE o2;

-- Create O2 user
CREATE USER o2 WITH PASSWORD 'changeme';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE o2 TO o2;

-- Switch to O2 database
\c o2;

-- Create tables for O2
CREATE TABLE IF NOT EXISTS proposals (
    id SERIAL PRIMARY KEY,
    room_id TEXT NOT NULL,
    author TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    proposal_type TEXT NOT NULL,
    threshold FLOAT NOT NULL,
    duration_hours INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ends_at TIMESTAMP NOT NULL,
    status TEXT DEFAULT 'active',
    result JSONB
);

CREATE TABLE IF NOT EXISTS votes (
    id SERIAL PRIMARY KEY,
    proposal_id INTEGER REFERENCES proposals(id),
    user_id TEXT NOT NULL,
    vote TEXT NOT NULL,
    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(proposal_id, user_id)
);

CREATE TABLE IF NOT EXISTS user_memories (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    encrypted_data BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, memory_id)
);

CREATE INDEX idx_proposals_room ON proposals(room_id);
CREATE INDEX idx_proposals_status ON proposals(status);
CREATE INDEX idx_votes_proposal ON votes(proposal_id);
CREATE INDEX idx_user_memories_user ON user_memories(user_id);
EOF

    echo -e "${GREEN}✓ PostgreSQL init script created${NC}"
else
    echo -e "${YELLOW}! init-db.sql already exists, skipping creation${NC}"
fi
echo ""

# Create Dockerfile for O2 bot
if [ ! -f Dockerfile ]; then
    echo "Creating Dockerfile..."
    cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy HoloLoom and O2 code
COPY ../HoloLoom /app/HoloLoom
COPY bot /app/bot
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create data directories
RUN mkdir -p /app/data /app/config /app/memories

# Health check endpoint
EXPOSE 8080

# Run O2 bot
CMD ["python", "-m", "bot.o2_bot"]
EOF

    echo -e "${GREEN}✓ Dockerfile created${NC}"
else
    echo -e "${YELLOW}! Dockerfile already exists, skipping creation${NC}"
fi
echo ""

# Create requirements.txt
if [ ! -f requirements.txt ]; then
    echo "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Matrix SDK
matrix-nio[e2e]==0.24.0

# Database
psycopg2-binary==2.9.9
redis==5.0.1

# HoloLoom dependencies (already in ../HoloLoom)
# Will be installed from local package

# LLM integrations
openai==1.3.0
anthropic==0.7.0

# Utilities
python-dotenv==1.0.0
aiohttp==3.9.0
asyncio==3.4.3
EOF

    echo -e "${GREEN}✓ requirements.txt created${NC}"
else
    echo -e "${YELLOW}! requirements.txt already exists, skipping creation${NC}"
fi
echo ""

# Start services
echo "Starting O2 services..."
echo ""

if [ "$PROFILE" = "production" ]; then
    echo "Starting in PRODUCTION mode (with Neo4j, Qdrant, Nginx)..."
    docker-compose --profile production up -d
else
    echo "Starting in DEVELOPMENT mode..."
    docker-compose up -d
fi

echo ""
echo "Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."
docker-compose ps

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}O2 Platform Setup Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# Load environment for displaying info
source .env

echo "Next steps:"
echo ""
echo "1. Register O2 bot user:"
echo "   docker exec -it o2-synapse register_new_matrix_user -c /data/homeserver.yaml http://localhost:8008"
echo "   Username: o2"
echo "   Password: (from .env: O2_BOT_PASSWORD)"
echo "   Admin: no"
echo ""
echo "2. Create a Matrix room and invite @o2:$MATRIX_SERVER_NAME"
echo ""
echo "3. Test O2:"
echo "   @o2 help"
echo "   @o2 what is platform anarchism?"
echo ""
echo "4. View logs:"
echo "   docker-compose logs -f o2-bot"
echo ""
echo "Services:"
echo "  - Synapse: http://localhost:8008"
echo "  - O2 Bot: http://localhost:8080 (health check)"
if [ "$PROFILE" = "production" ]; then
    echo "  - Neo4j: http://localhost:7474"
    echo "  - Qdrant: http://localhost:6333"
fi
echo ""
echo "Documentation:"
echo "  - Architecture: O2_PLATFORM_ARCHITECTURE.md"
echo "  - Manifesto: O2_MANIFESTO.md"
echo "  - User Guide: O2_USER_GUIDE.md"
echo ""
echo -e "${YELLOW}Remember: Platform Anarchism means YOU own your data!${NC}"
echo ""
