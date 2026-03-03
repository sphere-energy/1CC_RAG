# 1CC RAG Chatbot Production Readiness Plan

Last updated: 2026-02-19

## 1) Goal
Make the RAG chatbot production-ready for secure, reliable, observable, and scalable operation, with clear delivery criteria and verification gates.

## 2) Scope
This plan covers application code, runtime behavior, data layer, security, observability, testing, and deployment.

In scope files/components:
- `src/main.py`
- `src/chat/router.py`
- `src/chat/service.py`
- `src/chat/llm.py`
- `src/chat/retriever.py`
- `src/core/config.py`
- `src/core/auth.py`
- `src/core/database.py`
- `src/core/middleware.py`
- `src/limiter.py`
- `tests/test_chat.py`
- `docker-compose.yml`
- `Dockerfile`
- `alembic/versions/20251126_1222-ebd7d1cd3b60_initial_schema_with_users_conversations_.py`

## 3) Delivery Principles
- Blocker defects are fixed first (compilation, authentication, secrets).
- Security controls are enforced by default.
- Changes are accepted only with tests and measurable outcomes.
- No production rollout without rollback path and health validation.

## 4) Workstreams and Tasks

---

## WS-1: Blockers (Must complete before any release)

### T1. Fix prompt-construction syntax error
**Priority:** P0

**Specification**
- Correct invalid syntax in prompt construction in `src/chat/service.py`.
- Add unit tests for prompt assembly and ensure parser/type checks run in CI.

**Acceptance Criteria**
- Service imports without syntax errors.
- Chat endpoint can execute non-streaming and streaming paths.
- CI fails on syntax/type regressions for this module.

**Verification**
- Run static checks and tests.
- Run a local `/api/v1/chat` request with `stream=false` and `stream=true`.

---

### T2. Re-enable real authentication
**Priority:** P0

**Specification**
- Remove mock user and test-only bypass in `src/chat/router.py`.
- Restore dependency on `get_current_user` from `src/core/auth.py`.
- Ensure missing/invalid JWT returns 401 consistently.

**Acceptance Criteria**
- Requests without a valid token are rejected.
- Requests with a valid token resolve user claims correctly.
- No hardcoded test identity remains in runtime code.

**Verification**
- Integration tests for auth success/failure.
- Manual curl checks for 401/200 behavior.

---

### T3. Remove embedded secrets/default credentials
**Priority:** P0

**Specification**
- Eliminate insecure default DB credentials in `src/core/config.py`.
- Enforce mandatory env vars at startup for critical secrets.
- Document required environment variables in README/deploy docs.

**Acceptance Criteria**
- App fails fast with clear error if critical env vars are missing.
- No plaintext credentials remain in source-controlled defaults.
- Secrets are loaded from environment/secret manager only.

**Verification**
- Startup test with missing env vars.
- Security scan to detect hardcoded credentials.

---

## WS-2: Security and Compliance Hardening

### T4. Enforce authorization boundaries (tenant/user isolation)
**Priority:** P0

**Specification**
- Validate conversation ownership on every conversation access in `src/chat/service.py`.
- Ensure users cannot access/continue other users’ conversations.

**Acceptance Criteria**
- Cross-user conversation_id access is denied.
- Authorization checks are covered by tests.

**Verification**
- Integration tests with two users and shared DB fixtures.

---

### T5. Add prompt-injection and output safety controls
**Priority:** P1

**Specification**
- Add guardrails for untrusted user input in prompt construction.
- Add policy for handling uncertain legal answers (explicitly marked uncertainty).
- Add output validation layer before returning responses.

**Acceptance Criteria**
- Known injection payload test cases do not override system constraints.
- Unsafe patterns are blocked or sanitized.
- Safety behavior is documented and testable.

**Verification**
- Red-team test prompts and expected blocked/safe outcomes.

---

### T6. Reduce PII exposure in logs
**Priority:** P1

**Specification**
- Redact sensitive fields (email, token contents, message body where needed) in logs.
- Keep correlation IDs and operational metadata for debugging.

**Acceptance Criteria**
- No JWT, raw auth headers, or high-risk PII appears in logs.
- Logs still allow request tracing and incident triage.

**Verification**
- Log sampling under normal and error scenarios.

---

### T7. Harden CORS and API surface defaults
**Priority:** P1

**Specification**
- Restrict `allow_origins`, methods, and headers in `src/main.py` for production.
- Maintain environment-specific CORS configuration (dev vs prod).

**Acceptance Criteria**
- Production config only allows known frontend origins.
- Wildcard settings are not used in production.

**Verification**
- CORS preflight tests from allowed and disallowed origins.

---

### T8. Use user-aware rate limiting
**Priority:** P1

**Specification**
- Replace IP-only limiting in `src/limiter.py` with user/tenant-aware strategy when authenticated.
- Keep fallback keying for unauthenticated routes (if any).

**Acceptance Criteria**
- Limits apply fairly across NAT/shared IP environments.
- Abuse is throttled per user identity.

**Verification**
- Load tests with multiple users behind same IP.

---

## WS-3: Reliability and Performance

### T9. Reuse external clients instead of per-request creation
**Priority:** P0

**Specification**
- Stop constructing Bedrock and Qdrant clients per request in `src/chat/router.py`.
- Move client lifecycle to startup singletons/factories.

**Acceptance Criteria**
- Client instances are initialized once per process.
- Request latency decreases and connection churn is reduced.

**Verification**
- Benchmark cold/warm request latency before vs after.

---

### T10. Add robust retry/backoff/timeout behavior
**Priority:** P0

**Specification**
- Add explicit timeout, retry, exponential backoff, and jitter for Bedrock/Qdrant calls.
- Ensure circuit breaker behavior remains coherent with retry strategy.

**Acceptance Criteria**
- Transient 429/5xx errors are retried with bounded attempts.
- Permanent failures return stable API errors without hanging.

**Verification**
- Fault-injection tests for throttling/timeouts.

---

### T11. Define retrieval degradation strategy
**Priority:** P1

**Specification**
- Implement fallback behavior when retrieval fails (graceful user response vs hard failure).
- Include explicit metadata indicating degraded mode in response metadata.

**Acceptance Criteria**
- API remains responsive when Qdrant is partially unavailable.
- Degraded responses are identifiable for analytics.

**Verification**
- Simulate Qdrant downtime and observe behavior.

---

### T12. Add token/context guardrails
**Priority:** P1

**Specification**
- Add history truncation/summarization and max prompt size checks in `src/chat/service.py`.
- Limit runaway cost and request failure risk.

**Acceptance Criteria**
- Large histories do not exceed model input limits.
- Token usage remains within configured budget.

**Verification**
- Tests with oversized conversations and long retrieved context.

---

## WS-4: Data Model, Persistence, and Governance

### T13. Improve DB indexing and query performance
**Priority:** P1

**Specification**
- Review and add indexes for common query patterns (conversation and message retrieval).
- Add migration(s) with measured impact.

**Acceptance Criteria**
- Key read paths improve p95 query latency.
- Migrations are backward-compatible and reversible.

**Verification**
- EXPLAIN ANALYZE before/after on representative queries.

---

### T14. Ensure transactional consistency and idempotency
**Priority:** P0

**Specification**
- Define clear transaction boundaries around user/assistant message writes.
- Avoid partial writes in stream interruption/failure scenarios.

**Acceptance Criteria**
- No orphan or half-committed records in failure tests.
- Retry-safe behavior for duplicate submission scenarios.

**Verification**
- Failure-mode tests during stream and DB interruptions.

---

### T15. Implement retention and delete-by-user controls
**Priority:** P1

**Specification**
- Define data retention period and archival/deletion jobs.
- Provide API/mechanism to erase user data on request.

**Acceptance Criteria**
- Retention policy documented and enforced.
- User data deletion is complete, auditable, and tested.

**Verification**
- End-to-end deletion test and audit log verification.

---

## WS-5: Observability and Operations

### T16. Add structured metrics and tracing
**Priority:** P0

**Specification**
- Capture metrics for request latency, Bedrock/Qdrant call latency/error rates, circuit state, and retrieval quality signals.
- Keep correlation ID propagation end-to-end.

**Acceptance Criteria**
- Dashboards show p50/p95/p99 latency and dependency health.
- Alerts trigger on SLO violations and dependency failure spikes.

**Verification**
- Synthetic traffic and alert fire-drill.

---

### T17. Define SLOs and alert thresholds
**Priority:** P1

**Specification**
- Establish service SLOs for availability, latency, and error rate.
- Map alerts to runbooks with clear owner/escalation.

**Acceptance Criteria**
- SLOs published and agreed by engineering/product.
- Alert rules exist and are validated.

**Verification**
- Runbook tabletop exercise.

---

## WS-6: Testing and Quality Gates

### T18. Expand tests beyond mocked happy paths
**Priority:** P0

**Specification**
- Extend `tests/test_chat.py` to cover auth, authorization, DB persistence, streaming, and failure paths.
- Add integration tests for external dependency failures.

**Acceptance Criteria**
- Core flows (success + failure) are covered by automated tests.
- Regressions in auth/data isolation are caught by CI.

**Verification**
- CI test run with coverage reports and gating thresholds.

---

### T19. Add CI quality/security gates
**Priority:** P0

**Specification**
- Enforce linting, type checking, tests, dependency scanning, and container vulnerability scanning.
- Block merges on failed gates.

**Acceptance Criteria**
- CI pipeline fails on code quality/security violations.
- Artifacts include scan reports.

**Verification**
- Intentional failing PR validates gate behavior.

---

## WS-7: Deployment and Runtime Packaging

### T20. Fix compose/runtime inconsistencies
**Priority:** P0

**Specification**
- `docker-compose.yml` references `qdrant` in `depends_on` but does not define it.
- Add proper service definition (or remove dependency if externalized) with clear environment strategy.

**Acceptance Criteria**
- `docker compose up` works deterministically in local/dev.
- Service health checks pass for all required dependencies.

**Verification**
- Fresh environment bring-up test.

---

### T21. Make builds reproducible and secure
**Priority:** P1

**Specification**
- Pin dependencies (or lock file approach), minimize image footprint, and scan image vulnerabilities.
- Ensure production image uses least-privilege runtime and secure defaults.

**Acceptance Criteria**
- Rebuilds are deterministic.
- No critical container vulnerabilities at release time.

**Verification**
- Repeatable build checks and security scan reports.

---

## WS-8: Top-Tier Chatbot Capabilities (Product Excellence)

### T22. Hybrid memory architecture (short-term + long-term)
**Priority:** P1

**Specification**
- Implement layered memory: session memory (recent turns), episodic memory (conversation summaries), and user profile memory (stable preferences).
- Store long-term memory with explicit metadata (`memory_type`, `source`, `confidence`, `created_at`, `expires_at`).
- Use retrieval-time memory selection so only relevant memory is injected per query.

**Acceptance Criteria**
- The assistant consistently recalls user-specific stable preferences across sessions.
- Irrelevant past context is not injected into unrelated requests.
- Memory records are auditable and support TTL/expiration.

**Verification**
- Multi-session tests validating recall and non-recall cases.
- Memory retrieval precision/recall checks on curated scenarios.

---

### T23. Conversation history intelligence (compression + salience)
**Priority:** P1

**Specification**
- Add history compaction with salience scoring (facts, decisions, constraints, unresolved questions).
- Persist rolling summaries and regenerate them every N turns or token threshold.
- Keep raw transcript + summary pointers for traceability.

**Acceptance Criteria**
- Long conversations maintain answer quality without context overflow.
- Summaries preserve key commitments and legal references.
- Token usage remains bounded as conversations grow.

**Verification**
- Regression set with 50+ turn dialogs.
- Quality comparison (full history vs summarized history) within defined delta.

---

### T24. Query understanding and routing
**Priority:** P1

**Specification**
- Add query classification: legal lookup, follow-up clarification, procedural guidance, out-of-domain.
- Route to specialized prompt/retrieval templates based on intent.
- Enforce out-of-domain handling with safe refusal + redirect guidance.

**Acceptance Criteria**
- Intent routing accuracy meets agreed threshold on evaluation dataset.
- Out-of-domain prompts are handled safely and consistently.

**Verification**
- Labeled intent benchmark tests.
- Safety tests for adversarial/off-domain prompts.

---

### T25. Retrieval quality uplift (hybrid + rerank + citations)
**Priority:** P1

**Specification**
- Implement hybrid retrieval (vector + keyword/BM25) and reranking.
- Attach source-grounded citations to generated claims with traceable chunk IDs.
- Add retrieval diagnostics (`retrieved_k`, `rerank_scores`, `citation_coverage`).

**Acceptance Criteria**
- Measurable improvement in grounded answer quality.
- Responses include verifiable citations for legal claims.
- Hallucinated uncited legal claims are reduced below target threshold.

**Verification**
- Offline RAG evaluation (precision@k, MRR, groundedness score).
- Manual legal QA audit on sampled responses.

---

### T26. Personalization controls and memory governance
**Priority:** P1

**Specification**
- Add user controls: view memory, edit memory, forget memory, and disable personalization.
- Add per-tenant policy for what can/cannot be stored as memory.
- Log all memory mutations for auditability.

**Acceptance Criteria**
- Users can fully inspect and remove stored memory.
- Memory retention aligns with policy and compliance settings.
- Personalization can be disabled without breaking core chat.

**Verification**
- End-to-end tests for memory CRUD and opt-out behavior.
- Audit log review for all memory changes.

---

### T27. Multi-turn tool use and workflow orchestration
**Priority:** P2

**Specification**
- Add deterministic workflow support for multi-step tasks (e.g., compare regulations, build compliance checklist).
- Define tool contracts with strict schema validation and retry semantics.
- Persist intermediate reasoning artifacts as structured workflow state (not raw chain-of-thought).

**Acceptance Criteria**
- Complex requests complete reliably across multiple tool steps.
- Tool errors are recoverable with transparent user messaging.

**Verification**
- Scenario tests for multi-step legal workflows.
- Fault injection on tool failures and retries.

---

### T28. UX quality features for premium chat experience
**Priority:** P2

**Specification**
- Add streaming improvements: sectioned responses, progress states, and partial citation updates.
- Add follow-up suggestion generation based on unresolved user intent.
- Add explicit confidence and uncertainty cues where evidence is weak.

**Acceptance Criteria**
- Users receive clear progressive feedback during long responses.
- Suggested follow-ups are relevant and increase successful task completion.
- Uncertain answers are labeled and include next-best actions.

**Verification**
- UX telemetry for abandonment and completion rates.
- A/B test against baseline chat flow.

---

### T29. Evaluation framework for chatbot excellence
**Priority:** P0

**Specification**
- Establish an automated eval suite: factuality, groundedness, legal citation quality, safety, latency, and personalization correctness.
- Version prompts/retrievers/models and track score deltas per release.
- Add release gate thresholds for critical quality metrics.

**Acceptance Criteria**
- Every release includes eval report and pass/fail decision.
- Regressions in factuality/safety block promotion.

**Verification**
- CI/CD integration of eval pipeline.
- Historical dashboard of quality trends by release.

---

## 5) Non-Functional Targets (Initial)
- API availability: >= 99.9%
- API p95 latency (non-streaming): <= 3.0s under agreed baseline load
- 5xx error rate: < 1% steady state
- Authn/Authz bypass incidents: 0
- Data isolation violations: 0

(These targets should be validated and finalized with product + SRE.)

## 6) Definition of Done (Global)
A task is considered done only when all are true:
1. Code implemented and merged.
2. Automated tests added/updated and passing.
3. Observability updated where relevant (logs/metrics/traces).
4. Documentation/runbook updated.
5. Security/privacy implications reviewed.

## 7) Week-by-Week Execution Plan (Updated Priority)

### Week 1 (Essential for platform correctness)
1. WS-1 Blockers (T1-T3)
2. WS-8 Top-Tier Chatbot Capabilities (T22-T29) — full scope now mandatory in Week 1
3. WS-2 critical security controls (T4, T7, T8)
4. WS-3 reliability core (T9, T10)
5. WS-6 quality gate foundation (T18, T19)

### Week 2
1. WS-4 data consistency and governance (T14, T15, T13)
2. WS-5 observability hardening (T16, T17)

### Week 3
1. WS-7 deployment/runtime packaging hardening (T20, T21)
2. Performance tuning and stabilization pass across all Week 1 and Week 2 deliverables

Implementation note: WS-8 items are treated as core product functionality, not optional enhancements.

## 8) Release Readiness Gate
Do not promote to production until:
- P0 tasks are complete and verified.
- No open critical security findings.
- Quality eval suite (T29) passes release thresholds.
- SLO dashboards and alerting are live.
- Rollback plan is tested.
- Sign-off obtained from engineering owner + security owner.
