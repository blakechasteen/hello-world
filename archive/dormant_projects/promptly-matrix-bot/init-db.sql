-- Promptly Database Initialization
-- Creates the promptly database and user for the bot

-- Create promptly user (if not exists)
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'promptly') THEN

      CREATE ROLE promptly LOGIN PASSWORD 'changeme';
   END IF;
END
$do$;

-- Create promptly database
CREATE DATABASE promptly
    WITH
    OWNER = promptly
    ENCODING = 'UTF8'
    LC_COLLATE = 'C'
    LC_CTYPE = 'C'
    TEMPLATE = template0;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE promptly TO promptly;