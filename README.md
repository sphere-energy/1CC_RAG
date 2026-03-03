# 1CC RAG API - Production-Grade Legal Assistant

A production-ready Retrieval-Augmented Generation (RAG) API for answering questions about European electronics and battery legislation. Built for **Sphere Energy 1CC** consulting services.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Overview

This API provides an intelligent legal consultant assistant (Liggy) that answers questions about EU legislation regarding electronics and batteries. It uses:

- **Retrieval-Augmented Generation (RAG)** - Combines vector search with Large Language Models
- **AWS Bedrock** - Claude 3.5 Sonnet for text generation, Cohere for embeddings
- **Qdrant** - Vector database for legislation documents
- **AWS Cognito** - Enterprise authentication and user management
- **PostgreSQL (AWS RDS)** - Conversation and message persistence
- **Circuit Breakers** - Fault-tolerant architecture

### Key Features

✅ **Enterprise Authentication** - AWS Cognito JWT verification  
✅ **Conversation History** - Full multi-turn conversation support with database persistence  
✅ **Streaming Responses** - Real-time Server-Sent Events for immediate feedback  
✅ **Circuit Breakers** - Resilient against external service failures  
✅ **Observability** - Structured JSON logging with correlation IDs  
✅ **Production Ready** - CORS, rate limiting, compression, health checks  
✅ **Database Migrations** - Alembic for schema version control  
✅ **Dockerized** - Ready for container deployment

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16+ (or AWS RDS)
- AWS Account with:
  - Cognito User Pool
  - RDS PostgreSQL instance
  - IAM permissions for Bedrock
- Access to Qdrant vector database

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd 1CC_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### Environment Configuration

Edit `.env` with your values:

```bash
# Runtime mode
ENVIRONMENT=dev  # dev | test | prod

# AWS Cognito
COGNITO_REGION=eu-central-1
COGNITO_USER_POOL_ID=eu-central-1_XXXXXXXXX
COGNITO_CLIENT_ID=your-client-id  # Optional

# AWS RDS PostgreSQL
DATABASE_URL=postgresql://username:password@your-rds.eu-central-1.rds.amazonaws.com:5432/rag_db

# AWS Credentials (for Bedrock)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-central-1

# Qdrant
QDRANT_HOST=3.78.186.81
QDRANT_PORT=6333

# Security / runtime controls
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
CORS_ALLOW_METHODS=GET,POST,OPTIONS
CORS_ALLOW_HEADERS=Authorization,Content-Type,X-Request-ID
RATE_LIMIT=5/minute
```

Critical startup requirement: the application fails fast if `DATABASE_URL` or `COGNITO_USER_POOL_ID` is missing.

### Database Setup

```bash
# Create initial migration
alembic revision --autogenerate -m "Initial schema with users, conversations, messages"

# Apply migrations
alembic upgrade head

# Verify tables created
psql $DATABASE_URL -c "\dt"
```

### Run the Application

**Development:**
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

For local curl verification without Cognito token (temporary testing mode):
```bash
ALLOW_UNAUTHENTICATED_REQUESTS=true uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn src.main:app --workers 4 --host 0.0.0.0 --port 8000
```

**Docker:**
```bash
docker-compose up --build
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "ok", "version": "1.0.0-production", "service": "1CC RAG API"}

# View API documentation
open http://localhost:8000/docs
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────┐      HTTPS/TLS       ┌──────────────────┐
│   Client    │ ────────────────────▶│  AWS ALB (TLS)   │
│ (Frontend)  │                       └──────────────────┘
└─────────────┘                                │
                                               ▼
                    ┌────────────────────────────────────────────┐
                    │         FastAPI Application (ECS)          │
                    │  ┌──────────────────────────────────────┐  │
                    │  │  Middleware Stack                    │  │
                    │  │  - Correlation ID                    │  │
                    │  │  - Request Logging                   │  │
                    │  │  - Rate Limiting (SlowAPI)           │  │
                    │  │  - CORS                              │  │
                    │  │  - Brotli Compression                │  │
                    │  └──────────────────────────────────────┘  │
                    │                    │                        │
                    │  ┌──────────────────────────────────────┐  │
                    │  │  Authentication Layer                │  │
                    │  │  - AWS Cognito JWT Verification      │  │
                    │  │  - User Creation/Lookup              │  │
                    │  └──────────────────────────────────────┘  │
                    │                    │                        │
                    │  ┌──────────────────────────────────────┐  │
                    │  │  Business Logic (ChatService)        │  │
                    │  │  - Conversation Management           │  │
                    │  │  - Message Persistence               │  │
                    │  │  - RAG Orchestration                 │  │
                    │  └──────────────────────────────────────┘  │
                    │         │              │           │        │
                    └─────────┼──────────────┼───────────┼────────┘
                              │              │           │
                ┌─────────────┘              │           └────────────┐
                │                            │                        │
         Circuit Breaker              Circuit Breaker          PostgreSQL
                │                            │                  (AWS RDS)
                ▼                            ▼                        │
        ┌──────────────┐            ┌──────────────┐          ┌─────────────┐
        │ AWS Bedrock  │            │   Qdrant     │          │   Tables    │
        │              │            │  (Vector DB) │          │  - users    │
        │ - Claude 3.5 │            │              │          │  - convs    │
        │ - Cohere     │            │  Legislation │          │  - messages │
        └──────────────┘            └──────────────┘          └─────────────┘
```

### Component Breakdown

#### 1. **API Layer** (`src/main.py`, `src/chat/router.py`)
- FastAPI application with automatic OpenAPI docs
- RESTful endpoints with proper HTTP semantics
- Dependency injection for clean architecture
- Exception handlers for structured error responses

#### 2. **Authentication** (`src/core/auth.py`)
- AWS Cognito integration via JWT tokens
- Automatic JWKS fetching and caching
- Token signature & claims verification
- User extraction from JWT (sub, email, username)

#### 3. **Database Layer** (`src/core/database.py`, `src/chat/models.py`)
- SQLAlchemy ORM for type-safe database operations
- Connection pooling optimized for AWS RDS
- Automatic user creation from Cognito
- Conversation threading with message history
- JSONB metadata storage for sources & analytics

#### 4. **RAG Pipeline** (`src/chat/service.py`)
```
User Query
    ↓
Generate Embedding (BedrockClient)
    ↓
Vector Search (QdrantRetriever)
    ├─ Macro chunks (legislation sections)
    ├─ Micro chunks (detailed paragraphs)
    └─ Context expansion (neighboring chunks)
    ↓
Format Context + History
    ↓
Generate Response (BedrockClient + Claude)
    ↓
Save to Database
    ↓
Return Response
```

#### 5. **Circuit Breakers** (`src/core/circuit_breaker.py`)
- **Bedrock Breaker**: Protects against AWS Bedrock outages
- **Qdrant Breaker**: Protects against vector DB failures
- Prevents cascading failures
- Automatic recovery after timeout period

#### 6. **Observability** (`src/core/middleware.py`)
- Correlation IDs for distributed tracing
- Structured JSON logging (compatible with CloudWatch, Datadog)
- Request/response duration tracking
- Error context preservation

---

## 📡 API Usage

### Authentication

All endpoints require AWS Cognito JWT token in the `Authorization` header:

```bash
Authorization: Bearer eyJraWQiOiJ...
```

### Endpoints

#### 1. POST `/api/v1/chat` - Chat with Liggy

**Request:**
```json
{
  "conversation_id": "uuid-optional",  // Omit for new conversation
  "messages": [
    {
      "role": "user",
      "content": "What are the Danish regulations for electronic waste disposal?"
    }
  ],
  "stream": false
}
```

**Response (Non-Streaming):**
```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "660e8400-e29b-41d4-a716-446655440001",
  "response": "Based on the Danish legislation regarding electronic waste (WEEE Directive implementation)...\n\n**Key Requirements:**\n- Collection rate: 65% of electronics sold\n- Producer responsibility for recycling costs\n- Consumer drop-off points within 15km...\n\n**Sources:**\n- Danish WEEE Act 2023, §4.2\n- EU Directive 2012/19/EU, Article 7"
}
```

**Response (Streaming with `stream: true`):**
```
event: conversation_id
data: 550e8400-e29b-41d4-a716-446655440000

data: Based on the

data: Danish legislation

data: regarding electronic waste...
```

#### 2. GET `/health` - Health Check

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0-production",
  "service": "1CC RAG API",
  "correlation_id": "abc-123-def-456"
}
```

### Error Responses

All errors follow a consistent structure:

```json
{
  "error_type": "bedrock_error",
  "message": "Rate limit exceeded. Please try again later.",
  "detail": {
    "error_code": "ThrottlingException",
    "error_message": "Rate exceeded"
  },
  "correlation_id": "abc-123"
}
```

**Error Types:**
- `validation_error` - Invalid request format (422)
- `bedrock_error` - AWS Bedrock issues (503)
- `qdrant_error` - Vector database issues (503)
- `auth_error` - Authentication failures (401)
- `internal_server_error` - Unexpected errors (500)

### Rate Limiting

Default: **5 requests per minute** per IP address.

**Rate limit exceeded response:**
```json
{
  "error": "Rate limit exceeded: 5 per 1 minute"
}
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `COGNITO_REGION` | ✅ | `eu-central-1` | AWS region for Cognito |
| `COGNITO_USER_POOL_ID` | ✅ | - | Cognito User Pool ID |
| `COGNITO_CLIENT_ID` | ❌ | `None` | Optional client ID validation |
| `DATABASE_URL` | ✅ | - | PostgreSQL connection string |
| `DATABASE_POOL_SIZE` | ❌ | `10` | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | ❌ | `20` | Max overflow connections |
| `AWS_REGION` | ❌ | `eu-central-1` | AWS region for Bedrock |
| `BEDROCK_EMBEDDING_MODEL_ID` | ❌ | Cohere ARN | Embedding model |
| `BEDROCK_TEXT_MODEL_ID` | ❌ | Claude ARN | Text generation model |
| `QDRANT_HOST` | ❌ | `3.78.186.81` | Qdrant server host |
| `QDRANT_PORT` | ❌ | `6333` | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | ❌ | `1cc_legislation` | Collection name |
| `CIRCUIT_BREAKER_FAIL_MAX` | ❌ | `5` | Failures before opening circuit |
| `CIRCUIT_BREAKER_TIMEOUT` | ❌ | `60` | Seconds before retry (half-open) |
| `CORS_ORIGINS` | ❌ | `localhost:3000,8000` | Comma-separated allowed origins |
| `RATE_LIMIT` | ❌ | `5/minute` | Rate limit per IP |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level |

### Model Configuration

**Embedding Model:** Cohere Embed v4
- Dimension: 1024
- Input: Search queries
- Use case: Similarity search in legislation

**Text Generation Model:** Claude 3.5 Sonnet
- Context window: 200K tokens
- Max output: 10K tokens
- Temperature: 0.7 (balanced creativity/accuracy)

---

## 💻 Development

### Project Structure

```
src/
├── core/               # Infrastructure & cross-cutting concerns
│   ├── auth.py        # Cognito authentication
│   ├── circuit_breaker.py  # Fault tolerance
│   ├── config.py      # Settings management
│   ├── database.py    # PostgreSQL connection
│   ├── exceptions.py  # Custom exceptions
│   └──middleware.py  # Logging & correlation IDs
├── chat/              # Chat feature (package-by-feature)
│   ├── llm.py        # Bedrock client
│   ├── models.py     # Database models
│   ├── retriever.py  # Qdrant client
│   ├── router.py     # API endpoints
│   ├── schemas.py    # Request/response models
│   └── service.py    # Business logic
├── limiter.py        # Rate limiter instance
└── main.py           # Application entry point
```

### Running Tests

```bash
# Unit tests (mock external services)
pytest tests/test_chat.py -v

# All tests
pytest tests/ -v

# With coverage
pytest --cov=src tests/

# Integration tests (requires real services)
pytest tests/ --integration
```

### Code Quality

```bash
# Format code
black src tests

# Type checking
mypy src

# Linting
ruff check src
```

### Database Migrations

```bash
# Create migration after model changes
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# Current database version
alembic current
```

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build image
docker build -t 1cc-rag-api:latest .

# Run container
docker run -p 8000:8000 --env-file .env 1cc-rag-api:latest

# Docker Compose (includes Qdrant)
docker-compose up -d
```

### AWS ECS Deployment

1. **Push to ECR:**
```bash
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-central-1.amazonaws.com
docker tag 1cc-rag-api:latest <account-id>.dkr.ecr.eu-central-1.amazonaws.com/1cc-rag-api:latest
docker push <account-id>.dkr.ecr.eu-central-1.amazonaws.com/1cc-rag-api:latest
```

2. **Create ECS Task Definition** with:
   - Environment variables from `.env`
   - IAM role with Bedrock permissions
   - CloudWatch log group
   - Health check: `/health`

3. **Create ECS Service:**
   - Behind Application Load Balancer
   - Target group health check: `/health`
   - Auto-scaling based on CPU/memory

### Environment-Specific Configuration

**Development:**
```bash
LOG_LEVEL=DEBUG
DATABASE_ECHO=True  # Log SQL queries
```

**Staging:**
```bash
LOG_LEVEL=INFO
CIRCUIT_BREAKER_FAIL_MAX=3  # Faster failure detection
```

**Production:**
```bash
LOG_LEVEL=WARNING
DATABASE_POOL_SIZE=20  # Higher concurrency
RATE_LIMIT=100/minute  # Adjust based on load
```

---

## 📊 Monitoring

### Structured Logging

All logs are JSON formatted:

```json
{
  "timestamp": "2025-11-24T15:00:00Z",
  "level": "INFO",
  "logger": "src.chat.service",
  "message": "Processing chat request",
  "correlation_id": "abc-123-def-456",
  "user_sub": "cognito-user-sub",
  "conversation_id": "550e8400-...",
  "duration_ms": 1250
}
```

**CloudWatch Insights Queries:**

```sql
# Error rate
fields @timestamp, level, message
| filter level = "ERROR"
| stats count() by bin(5m)

# Slow requests
fields @timestamp, correlation_id, duration_ms
| filter duration_ms > 5000
| sort duration_ms desc

# Circuit breaker events
fields @timestamp, message
| filter message like /circuit.*open/
```

### Key Metrics to Monitor

- **Request latency** (p50, p95, p99)
- **Error rate** (by error_type)
- **Circuit breaker state** (open/closed events)
- **Database connection pool** usage
- **Token consumption** (cost tracking)
- **Active conversations** per user

---

## 🔧 Troubleshooting

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed troubleshooting guide.

**Common Issues:**

1. **"Invalid token"** → Check Cognito User Pool ID, token expiration
2. **"Circuit breaker open"** → Check Bedrock/Qdrant health, wait 60s
3. **"Database connection failed"** → Verify RDS security group, credentials
4. **Slow responses** → Check Bedrock throttling, Qdrant query performance

---

## 📄 License

This project is proprietary software developed for Sphere Energy 1CC.

---

## 👥 Support

For issues or questions:
- **Internal Wiki:** [Link to internal documentation]
- **Slack Channel:** #1cc-rag-api
- **On-Call:** [PagerDuty rotation]

---

## 🗺️ Roadmap

- [ ] **v1.1** - Caching layer (Redis) for frequent queries
- [ ] **v1.2** - User feedback collection & response ratings
- [ ] **v1.3** - Multi-language support (DE, FR, ES)
- [ ] **v2.0** - Multi-tenancy for different consulting clients
- [ ] **v2.1** - Advanced analytics dashboard