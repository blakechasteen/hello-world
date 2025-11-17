#!/bin/bash
# EdWIN Database Initialization Script
# Initializes databases with schema and seed data
#
# Usage:
#   ./scripts/init_db.sh [environment]
#
# Implementation Date: November 17, 2025

set -e

ENVIRONMENT="${1:-development}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}EdWIN Database Initialization - ${ENVIRONMENT}${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Wait for databases to be ready
echo -e "\n${YELLOW}Waiting for databases to be ready...${NC}"
sleep 10

# Initialize Neo4j indexes and constraints
echo -e "\n${YELLOW}Creating Neo4j indexes...${NC}"
if command -v docker &> /dev/null; then
    docker exec edwin-neo4j cypher-shell -u neo4j -p edwin_password_2025 \
        "CREATE INDEX user_email IF NOT EXISTS FOR (u:User) ON (u.email);"
    docker exec edwin-neo4j cypher-shell -u neo4j -p edwin_password_2025 \
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;"
    docker exec edwin-neo4j cypher-shell -u neo4j -p edwin_password_2025 \
        "CREATE INDEX student_id IF NOT EXISTS FOR (s:Student) ON (s.student_id);"
    docker exec edwin-neo4j cypher-shell -u neo4j -p edwin_password_2025 \
        "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name);"
    echo -e "${GREEN}✅ Neo4j indexes created${NC}"
fi

# Initialize Qdrant collections
echo -e "\n${YELLOW}Creating Qdrant collections...${NC}"
if command -v curl &> /dev/null; then
    # Create edwin collection for RAG
    curl -X PUT "http://localhost:6333/collections/edwin" \
        -H "Content-Type: application/json" \
        -d '{
          "vectors": {
            "size": 384,
            "distance": "Cosine"
          }
        }' > /dev/null 2>&1

    # Create embeddings collection
    curl -X PUT "http://localhost:6333/collections/embeddings" \
        -H "Content-Type: application/json" \
        -d '{
          "vectors": {
            "size": 384,
            "distance": "Cosine"
          }
        }' > /dev/null 2>&1

    echo -e "${GREEN}✅ Qdrant collections created${NC}"
fi

# Initialize curriculum
echo -e "\n${YELLOW}Loading curriculum data...${NC}"
PYTHONPATH=. python EduVerse/edwin/scripts/init_curriculum.py
echo -e "${GREEN}✅ Curriculum initialized${NC}"

# Create default users
echo -e "\n${YELLOW}Creating default users...${NC}"
python scripts/seed_data.py
echo -e "${GREEN}✅ Default users created${NC}"

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Database initialization complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
