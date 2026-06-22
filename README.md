# 1CC RAG API

A production RAG (Retrieval-Augmented Generation) API that answers questions about EU electronics and battery legislation. Built for **Sphere Energy 1CC** consulting services and powers the **Liggy** legal assistant.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-lightgrey.svg)](#license)

## Contents

- [What it does](#what-it-does)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Database Migrations](#database-migrations)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What it does

1. A user asks a legal question (e.g. *"What are the Danish WEEE recycling requirements?"*)
2. The query is embedded and matched against legislation stored in **Qdrant**
3. The question + retrieved context goes to **Claude** (via **AWS Bedrock**) to generate a grounded, cited answer
4. The conversation is saved to **PostgreSQL (AWS RDS)**
5. Every request is authenticated through **AWS Cognito**

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Generation model | Claude Haiku 4.5 (AWS Bedrock) |
| Embedding model | Cohere Embed v4 (AWS Bedrock) |
| Vector search | Qdrant |
| Database | PostgreSQL (AWS RDS) via SQLAlchemy + Alembic |
| Auth | AWS Cognito (JWT) |
| Resilience | Circuit breakers around Bedrock & Qdrant |

**Also included:** multi-turn conversation history, streaming responses (SSE), rate limiting, structured JSON logging with correlation IDs, and Docker support.

---

## Quick Start

### 1. Prerequisites
- Python 3.12+
- An AWS Cognito User Pool
- An AWS RDS PostgreSQL instance
- IAM permissions for Bedrock
- A reachable Qdrant instance

### 2. Install
```bash
git clone git@github.com:sphere-energy/1CC_RAG.git
cd 1CC_RAG
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```
Fill in the values — see [Configuration](#configuration) for the full reference. The app **fails to start** if `DATABASE_URL` or `COGNITO_USER_POOL_ID` is missing.

### 4. Run database migrations — always before starting the app
```bash
alembic current          # what revision is the DB on right now?
alembic heads            # what does the repo consider the latest revision?
alembic upgrade head     # apply everything in between
```
> Run these three commands, in this order, from the repo root, every time you pull new code or switch branches. Jumping straight to `upgrade head` without checking `current`/`heads` first is how mismatched-revision errors slip in — see [Database Migrations](#database-migrations) for what to do if they don't match.

### 5. Start the server
Activate the virtual environment first — every time you open a new terminal or come back to the project:
```bash
source venv/bin/activate     # Windows: venv\Scripts\activate
```
Then run one of:
```bash
# Development
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Local curl testing without a real Cognito token
ALLOW_MOCK_AUTH=true uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.main:app --workers 4 --host 0.0.0.0 --port 8000

# Docker (includes Qdrant — no venv needed)
docker-compose up --build
```

### 6. Verify
```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0-production", "service": "1CC RAG API"}

open http://localhost:8000/docs   # interactive API docs
```

---

## Architecture

```
Client → AWS ALB (TLS) → FastAPI app
           ├─ Middleware: correlation ID, logging, rate limiting, CORS, compression
           ├─ Auth: Cognito JWT verification
           └─ ChatService: conversation management + RAG orchestration
                ├─ AWS Bedrock   (circuit breaker) — Claude Haiku 4.5 + Cohere embeddings
                ├─ Qdrant        (circuit breaker) — legislation vector search
                └─ PostgreSQL (RDS) — users / conversations / messages
```

### Project structure
```
src/
├── core/        # auth, database, config, circuit breaker, middleware, exceptions
├── chat/        # llm client, models, retriever, router, schemas, service
├── limiter.py
└── main.py
alembic/
└── versions/    # one file per migration — always committed with its model change
```

---

## API Usage

Every endpoint requires `Authorization: Bearer <cognito-jwt>`.

### `POST /api/v1/chat`
```json
{
  "conversation_id": "uuid-optional",
  "messages": [{ "role": "user", "content": "What are the Danish WEEE rules?" }],
  "stream": false
}
```
- Omit `conversation_id` to start a new conversation.
- Set `"stream": true` to receive Server-Sent Events instead of one JSON response.

### `GET /health`
Returns status, version, and a correlation ID.

### Errors
Consistent shape — `error_type`, `message`, `detail`, `correlation_id`:

| `error_type` | HTTP status |
|---|---|
| `validation_error` | 422 |
| `auth_error` | 401 |
| `bedrock_error` | 503 |
| `qdrant_error` | 503 |
| `internal_server_error` | 500 |

### Rate limiting
Default **5 requests/minute** per IP (`RATE_LIMIT`).

---

## Configuration

`.env.example` — credentials below are placeholders, never commit real ones:

```dotenv
# Qdrant Configuration
QDRANT_HOST=3.78.186.81
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=1cc_legislation

# ENVIRONMENT=dev  # dev | test | prod

# AWS Configuration
# AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY
AWS_REGION=eu-central-1
BEDROCK_EMBEDDING_MODEL_ID=arn:aws:bedrock:eu-central-1:BEDROCK_EMBEDDING_MODEL_ID:inference-profile/eu.cohere.embed-v4:0
BEDROCK_TEXT_MODEL_ID=arn:aws:bedrock:eu-central-1:BEDROCK_TEXT_MODEL_ID:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0

# AWS Cognito Configuration
COGNITO_REGION=eu-central-1
COGNITO_USER_POOL_ID=eu-central-COGNITO_USER_POOL_ID
COGNITO_CLIENT_ID=COGNITO_CLIENT_ID

# Database Configuration (AWS RDS PostgreSQL)
DATABASE_URL=postgresql://postgres:DATABASE_PASSWORD@DATABASE_HOST:5432/DATABASE_NAME
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_ECHO=False

# Circuit Breaker Configuration
CIRCUIT_BREAKER_FAIL_MAX=5
CIRCUIT_BREAKER_TIMEOUT=60

# CORS Configuration (comma-separated list)
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
# CORS_ALLOW_METHODS=GET,POST,OPTIONS
# CORS_ALLOW_HEADERS=Authorization,Content-Type,X-Request-ID

# Rate Limiting
RATE_LIMIT=5/minute

# Logging
LOG_LEVEL=INFO
ALLOW_MOCK_AUTH=true

# Application
APP_NAME=1CC-RAG-API
APP_VERSION=1.0.0-production
```

### Reference

| Variable | Required | Default | Notes |
|---|---|---|---|
| `APP_NAME` / `APP_VERSION` | – | `1CC-RAG-API` / `1.0.0-production` | shown in `/health` and logs |
| `ENVIRONMENT` | – | `dev` | `dev` \| `test` \| `prod` |
| `LOG_LEVEL` | – | `INFO` | |
| `COGNITO_REGION` | ✅ | `eu-central-1` | |
| `COGNITO_USER_POOL_ID` | ✅ | – | app won't start without this |
| `COGNITO_CLIENT_ID` | – | none | optional extra token validation |
| `DATABASE_URL` | ✅ | – | app won't start without this |
| `DATABASE_POOL_SIZE` / `DATABASE_MAX_OVERFLOW` | – | `10` / `20` | |
| `DATABASE_ECHO` | – | `False` | dev-only SQL logging |
| `AWS_REGION` | – | `eu-central-1` | region for Bedrock |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | – | none | skip if using an IAM role/instance profile |
| `BEDROCK_EMBEDDING_MODEL_ID` | – | Cohere Embed v4 ARN | |
| `BEDROCK_TEXT_MODEL_ID` | – | Claude Haiku 4.5 ARN | generation model |
| `QDRANT_HOST` / `QDRANT_PORT` | – | `3.78.186.81` / `6333` | |
| `QDRANT_COLLECTION_NAME` | – | `1cc_legislation` | |
| `CIRCUIT_BREAKER_FAIL_MAX` | – | `5` | failures before the circuit opens |
| `CIRCUIT_BREAKER_TIMEOUT` | – | `60` | seconds before a half-open retry |
| `CORS_ORIGINS` | – | localhost:3000 / :8080 | JSON-array string |
| `RATE_LIMIT` | – | `5/minute` | per IP |
| `ALLOW_MOCK_AUTH` | – | `false` | **local dev only** — skips Cognito verification |

> Keep `.env` gitignored — verify with `git check-ignore -v .env` — and store the real `DATABASE_URL` and AWS credentials in a secrets manager (AWS Secrets Manager, 1Password), never in Slack, email, or this README.

---

## Database Migrations

The most common source of broken environments on this project is an applied-but-uncommitted migration. Follow this order and it won't happen to you.

### Day-to-day workflow
```bash
# 1. Edit models in src/chat/models.py
# 2. Generate the migration
alembic revision --autogenerate -m "add column X to table Y"
# 3. Read the generated file in alembic/versions/ before applying —
#    autogenerate misses index renames, sequence changes, etc.
# 4. Apply it
alembic upgrade head
# 5. Commit the model change and the migration file together, immediately
git add src/chat/models.py alembic/versions/
git commit -m "feat: add column X to table Y + migration"
```

**Golden rule:** a migration file and the model change it came from are a single atomic commit. Never end a session with an untracked file in `alembic/versions/`.

### Useful commands
| Command | Purpose |
|---|---|
| `alembic current` | revision the DB is on |
| `alembic heads` | latest revision known to the repo |
| `alembic history --verbose` | full migration chain |
| `alembic upgrade head` | apply all pending migrations |
| `alembic downgrade -1` | roll back one migration |
| `alembic stamp head` | mark the DB as up to date *without* running any DDL |
| `alembic stamp <revision>` | force the DB's tracked revision, no DDL — fails if the DB's *current* revision can't be resolved |
| `alembic stamp --purge <revision>` | same, but wipes `alembic_version` unconditionally first — use when the DB's current revision is itself a ghost/missing one |

### Before you diagnose: confirm `psql` is actually pointed at the real database

Alembic and `psql` get `DATABASE_URL` from two different places, and they can disagree without you noticing:

- **Alembic** reads it straight out of `.env` via `get_settings()` — it's always pointed at the right database, regardless of your shell.
- **`psql "$DATABASE_URL"`** reads it from your **shell environment**. If you've only ever put it in `.env` and never exported it, the variable is blank in your terminal — and `psql` silently falls back to your local default Postgres instead of erroring. You end up "confirming" something about a database you're not actually looking at.

Check this first, every time, before trusting a manual `psql` result:
```bash
echo $DATABASE_URL    # blank? psql will silently use your local default DB instead of RDS
```

If it's blank, export it from `.env` before running anything else:
```bash
set -a
source .env
set +a
echo $DATABASE_URL    # should now print your real RDS connection string
```
If `source .env` errors out (a value like `CORS_ORIGINS=["...","..."]` can trip a plain `source`), pull out just the one variable instead:
```bash
export DATABASE_URL=$(grep '^DATABASE_URL=' .env | cut -d '=' -f2-)
```

Only once `echo $DATABASE_URL` prints the real connection string should you run the diagnostics below.

### Fixing `Can't locate revision identified by 'XXXXXXXX'`
This error means Alembic can't find a revision file somewhere in the chain — but *where* it's missing from changes the fix, so check the real database first:

```bash
psql "$DATABASE_URL" -c "\dt"                              # does alembic_version exist at all?
psql "$DATABASE_URL" -c "SELECT * FROM alembic_version;"   # if it exists, what does the DB think it's on?
```

**Case A — `alembic_version` exists and points at the missing revision.**
A migration was applied directly to this database, but its file was never committed to git — typically because someone ran `alembic upgrade head` locally and moved on without an `alembic revision` commit alongside it.

```bash
ls alembic/versions/                                  # does the file exist locally?
git log --all --diff-filter=D -- "alembic/versions/*"  # was it deleted from git history?
grep -rn "<ghost_revision_id>" .                       # search the whole repo for it
```

- **Schema already correct** (the migration was a no-op, or the change is already live): create an empty placeholder file with that exact revision ID and leave `upgrade()`/`downgrade()` as `pass`:
  ```bash
  alembic revision --rev-id <ghost_revision_id> -m "reconcile applied migration"
  git add alembic/versions/
  git commit -m "fix: reconcile previously applied, uncommitted migration"
  alembic upgrade head   # now a clean no-op — DB is already at this revision
  ```
- **Real schema changes are missing**: recover or reconstruct the file from whoever ran it, commit it, then:
  ```bash
  alembic upgrade head
  ```
- **You created and applied that revision yourself, then abandoned it** (an experimental branch/migration you deleted on purpose and don't want back): there's nothing to reconcile or recover — you just want the DB to forget it ever happened and point at the revision you're actually keeping.

  First, make sure `DATABASE_URL` is actually exported in this shell before checking anything — if it's missing, `psql` silently checks your local default database instead of RDS and every conclusion below will be wrong:
  ```bash
  echo $DATABASE_URL                    # blank? export it:
  set -a; source .env; set +a           # (or: export DATABASE_URL=$(grep '^DATABASE_URL=' .env | cut -d '=' -f2-))
  ```
  Now check the live schema for leftovers from the abandoned migration:
  ```bash
  psql "$DATABASE_URL" -c "\dt"         # or \d <table> for the specific table(s) it touched
  ```
  If the schema is clean (matches what the migration you're keeping actually creates), try forcing the DB's tracked revision directly — this rewrites `alembic_version` only, no DDL runs:
  ```bash
  alembic stamp ebd7d1cd3b60      # use whatever your real `alembic heads` currently shows
  ```
  **Use `--purge` instead of plain `stamp` when that command fails with the same `Can't locate revision` error.** A plain `alembic stamp <revision>` still has to read and resolve the DB's *current* tracked revision before it writes the new one — so if that current value is itself the ghost revision (exactly this situation), `stamp` fails for the same reason `current`/`upgrade` did. `--purge` skips resolving the old value entirely: it unconditionally wipes the `alembic_version` table first, then writes the target revision.
  ```bash
  alembic stamp --purge ebd7d1cd3b60
  ```
  Either way, confirm it's fixed before moving on:
  ```bash
  alembic current                 # should now cleanly show ebd7d1cd3b60 (head)
  alembic upgrade head            # should be a clean no-op
  ```
  If the schema check above turned up orphaned columns/tables the abandoned migration actually created, don't stamp over them silently — drop them first (manually, or with a small new migration whose `upgrade()` reverts them) so the cleanup is tracked, *then* stamp.

**Case B — `alembic_version` doesn't exist at all (fresh/empty DB), but `upgrade head` still fails.**
The database isn't the problem — the migration *chain itself* is broken. Somewhere in `alembic/versions/`, a file's `down_revision` points to a revision ID with no matching file. Alembic has to resolve the entire chain before it can do anything with it, so a broken link blocks every command — `upgrade`, `downgrade`, and `revision --autogenerate` — not just the one you happened to run. This is also why you can't simply create a new migration your way out of it: autogenerate needs to find the current head first, and it can't.

```bash
# List every revision/down_revision pair in the repo
grep -n "^revision\|^down_revision" alembic/versions/*.py

# Did the missing id ever exist in git history at all?
git log --all --diff-filter=A -- "alembic/versions/*<missing_id>*"
```
- If the missing id never existed in git history, and the file pointing to it is your **first/root** migration (often named `..._initial_schema_...`), the reference is simply wrong — there was never an earlier migration for it to depend on. Fix it by setting `down_revision = None` in that file, making it the root of the chain.
- If the missing id *did* exist at some point (the `git log` above shows it was added then deleted), recover that file instead of re-rooting — re-rooting would silently skip whatever schema it was responsible for.
- Re-verify the chain before touching the database again:
```bash
alembic heads              # exactly one head, no errors
alembic history --verbose  # one unbroken line, no errors
```
Only once both resolve cleanly should you run `alembic upgrade head` or `alembic revision --autogenerate`. On a genuinely empty DB, `upgrade head` will then create `alembic_version` from scratch and apply every migration in order.

### Team rules
1. Commit migration files immediately — `revision` → `git add` → `git commit`, one step.
2. Never run migrations against a shared/prod DB from a dirty working tree (`git status` first).
3. Always run `alembic current` + `alembic heads` before `alembic upgrade head`.
4. Never edit or delete a migration that's already been applied to a shared DB.
5. Keep `.env` gitignored; the real `DATABASE_URL` lives in the secrets manager only.
6. If you ever hand-edit a `revision` or `down_revision` id, immediately run `alembic heads` and `alembic history --verbose` to confirm the chain still resolves — a broken link stops the whole team from generating or applying migrations, not just you.
7. Before trusting any manual `psql "$DATABASE_URL"` check, run `echo $DATABASE_URL` first — if it's blank, `psql` is silently talking to your local default database, not RDS, and whatever it tells you is meaningless.

---

## Development

```bash
pytest tests/ -v              # unit tests (external services mocked)
pytest --cov=src tests/       # with coverage
pytest tests/ --integration   # against real services

black src tests               # format
ruff check src                # lint
mypy src                      # type check
```

`get_settings()` is `@lru_cache`'d — call `get_settings.cache_clear()` in test setup whenever a test overrides env vars, or you'll read stale settings.

---

## Deployment

### Docker
```bash
docker build -t 1cc-rag-api:latest .
docker run -p 8000:8000 --env-file .env 1cc-rag-api:latest
```

### AWS ECS
```bash
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-central-1.amazonaws.com
docker tag 1cc-rag-api:latest <account-id>.dkr.ecr.eu-central-1.amazonaws.com/1cc-rag-api:latest
docker push <account-id>.dkr.ecr.eu-central-1.amazonaws.com/1cc-rag-api:latest
```
Then create a Task Definition (env vars from `.env`, IAM role with Bedrock permissions, CloudWatch log group, health check `/health`) and a Service behind an ALB with the target group health check pointed at `/health`.

| Environment | Log level | Notes |
|---|---|---|
| Dev | `DEBUG` | `DATABASE_ECHO=True` to see SQL |
| Staging | `INFO` | `CIRCUIT_BREAKER_FAIL_MAX=3` for faster failure detection |
| Prod | `WARNING` | `DATABASE_POOL_SIZE=20`, raise `RATE_LIMIT` to real traffic needs |

---

## Troubleshooting

Logs are structured JSON with a `correlation_id` you can grep across a request's entire lifecycle — CloudWatch/Datadog compatible.

| Symptom | Likely cause | Fix |
|---|---|---|
| `Can't locate revision 'XXXXXXXX'` | either a migration applied to the DB was never committed, **or** the local migration chain itself has a broken `down_revision` link | check whether `alembic_version` exists in the DB first — see [Database Migrations](#database-migrations), the fix differs for each case |
| `Field required: database_url` | `DATABASE_URL` missing from `.env` | add it, confirm with `grep DATABASE_URL .env`, and run alembic from the repo root |
| `psql $DATABASE_URL` connects to the wrong DB, or shows tables/errors that don't match what Alembic reports | `DATABASE_URL` is only in `.env`, never exported to your shell, so `psql` silently falls back to your local default DB | `echo $DATABASE_URL` before trusting any `psql` output; if blank, run `set -a; source .env; set +a` — see [Database Migrations](#database-migrations) |
| "Invalid token" | wrong/expired Cognito token or wrong pool | check `COGNITO_USER_POOL_ID` and the token's issuer |
| "Circuit breaker open" | Bedrock or Qdrant unhealthy | check service health; the breaker auto-resets after `CIRCUIT_BREAKER_TIMEOUT` seconds |
| Slow responses | Bedrock throttling or Qdrant query latency | check CloudWatch; raise `external_timeout_seconds` if needed |
| Stale settings in tests | `get_settings()` is `@lru_cache`'d | call `get_settings.cache_clear()` in test setup |

---

## License

Proprietary software developed for Sphere Energy 1CC. Questions go to **#1cc-rag-api**.