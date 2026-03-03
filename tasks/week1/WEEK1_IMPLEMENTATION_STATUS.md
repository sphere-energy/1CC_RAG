# Week 1 Implementation Status

This file tracks concrete implementation artifacts for Week 1 items from `PRODUCTION_READINESS_TASKS.md`.

## WS-1 Blockers
- T1 prompt syntax and prompt assembly fixed in `src/chat/service.py`.
- T2 auth re-enabled in `src/chat/router.py` and `src/core/auth.py`.
- T3 critical secrets enforced in `src/core/config.py` and documented in `README.md`.

## WS-2 Security Controls
- T4 conversation ownership enforcement in `src/chat/service.py` (`_resolve_conversation`).
- T7 production-safe CORS configuration in `src/main.py` and `src/core/config.py`.
- T8 user-aware limiter keying in `src/limiter.py`.

## WS-3 Reliability Core
- T9 singleton Bedrock/Qdrant clients in `src/chat/router.py`.
- T10 timeout/retry/backoff/jitter in `src/chat/llm.py` and `src/chat/retriever.py`.

## WS-6 Quality Gates Foundation
- T18 expanded endpoint and failure-path tests in `tests/test_chat.py`.
- T19 CI lint/type/test/security gates in `.github/workflows/ci.yml`.

## WS-8 Top-Tier Capabilities (MVP delivery)
- T22 memory architecture primitives + persisted profile memory controls in `src/chat/service.py` and `src/chat/router.py`.
- T23 history salience and summarization in `src/chat/service.py`.
- T24 intent classification and routing in `src/chat/service.py`.
- T25 hybrid reranking/citations diagnostics in `src/chat/retriever.py` and `src/chat/service.py`.
- T26 memory governance endpoints in `src/chat/router.py` (`/memory*`).
- T27 deterministic workflow state metadata in `src/chat/service.py`.
- T28 streaming progress and metadata events in `src/chat/service.py` + `src/chat/router.py`.
- T29 evaluation scaffold in `src/chat/evaluation.py` and `tests/test_evaluation.py`.
