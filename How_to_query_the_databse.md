Option 1: From Host Machine:
bash# Install PostgreSQL client
brew install postgresql  # macOS
sudo apt install postgresql-client  # Ubuntu

# Connect to database
psql postgresql://litellm_user:postgres_password@localhost:5432/litellm_db

# Query logs
SELECT
    model,
    COUNT(*) as requests,
    SUM(total_tokens) as total_tokens,
    SUM(spend) as total_cost
FROM "LiteLLM_SpendLogs"
WHERE startTime >= NOW() - INTERVAL '1 day'
GROUP BY model;
Option 2: From Docker:
bash# Enter PostgreSQL container
docker exec -it stream-postgres psql -U litellm_user -d litellm_db

# Query
SELECT * FROM "LiteLLM_SpendLogs" LIMIT 10;



------------------------

# Connect to your PostgreSQL
docker exec -it stream-postgres psql -U litellm_user -d litellm_db

# List all tables
\dt

# You'll see:
                   List of relations
 Schema |            Name             | Type  |    Owner
--------+-----------------------------+-------+--------------
 public | LiteLLM_SpendLogs          | table | litellm_user
 public | LiteLLM_VerificationToken  | table | litellm_user
 public | LiteLLM_Config             | table | litellm_user
