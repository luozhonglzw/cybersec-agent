# CyberSec Agent

[![CI](https://github.com/luozhonglzw/cybersec-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/luozhonglzw/cybersec-agent/actions/workflows/ci.yml)
[**Release v0.5.0**](https://github.com/luozhonglzw/cybersec-agent/releases/tag/v0.5.0) · [MIT License](LICENSE) · Python 3.12+ · LangGraph · FastAPI

A security-operations agent prototype built on LangGraph, with a deliberate boundary between untrusted model output and authoritative planning, policy, and audit behavior.

**Current release:** v0.5.0 (2026-10-24) — knowledge-base Q&A over a version-controlled corpus, with three explicitly selected retrieval backends and structural citation validation.

## Run the offline demo (no LLM provider required)

```bash
uv run python scripts/demo_offline.py
```

Four scenarios, each computed from a real return value — no API key, no network, no Docker. First time here? [Quick Start](#quick-start) covers environment setup; this section deliberately does not repeat it.

| Scenario | Kind | What it demonstrates |
| --- | --- | --- |
| `DEMO 1` RBAC | real HTTP through the real FastAPI app | 401 / 403; unknown and missing credentials are byte-identical |
| `DEMO 2` Citation validation | real corpus, real retrieval, real validator | fabricated citations → `grounded=false` |
| `DEMO 3` Retrieval fail-closed | real Qdrant Local assembly path | broken index → `status="unavailable"`, no fallback |
| `DEMO 4` HITL | **test-based** (real pytest subprocess) | pause / resume / thread isolation / approver constraints |

`DEMO 2`'s answer text is **scripted** — there is no provider in this run — but the retrieval, the evidence envelope and the citation validator are real code. `DEMO 4` is a **test-based** demonstration, not a live `/triage → /resume` round trip: that path enters the graph at the agent node, so it needs a reachable provider. The script exits non-zero if any scenario fails.

Output of a real local run at `b3e7950`:

```text
  PASS  DEMO 1  RBAC — authentication and role admission
  PASS  DEMO 2  Citation validation — fabricated citations are rejected
  PASS  DEMO 3  Retrieval fail-closed — a broken index is refused, not replaced
  PASS  DEMO 4  HITL approval and resume — TEST-BASED DEMONSTRATION

DEMO PASSED: 4/4 scenarios
```

## Verified by CI

| Job | Scope | Observed at `b3e7950` |
| --- | --- | --- |
| `test` | offline suite — no network, no API key | **2,283 passed**, then the 4/4 demo |
| `test-postgres` | PostgreSQL integration against a throwaway server | **373 passed, 0 skipped** |
| `docker` | real image build + corpus load + route check | image built; 40 docs / 80 chunks in-container |

Run [`40114943132`](https://github.com/luozhonglzw/cybersec-agent/actions/runs/40114943132), commit `b3e7950`. That is one run on one commit against one throwaway server — not a standing guarantee for later commits (see [Known Limitations](#known-limitations)).

## Core design highlights

1. **The model cannot reach the safety path — structurally, not by convention.** The `plan` and `policy_gate` nodes read only the caller-supplied indicator and the deterministic plan; neither reads the message history. A CI assertion parses `app/core/graph.py` and fails if the identifier `messages` appears in either function body. This is a regression constraint over one syntactic property, not a formal proof of security.
2. **The policy gate is a graph node, not a tool.** A tool is optional for the model to call, which would make policy advisory; a node sits on the only path.
3. **Append-only is enforced by the database, not by application discipline.** 8 anti-mutation triggers on SQLite and 12 on PostgreSQL; on PostgreSQL the runtime role holds no DDL and cannot disable the triggers, because that requires table ownership. Pending approval state is derived by querying for the absence of a terminal event — there is no mutable status column.
4. **Citations are validated structurally.** Evidence is collected only from messages of type `tool` whose content is a valid `rag_hits` envelope, and the model can only produce `ai` messages — so a forged tool result is not reachable. Conflicting metadata for the same citation id fails closed for the whole registry.
5. **Three retrieval backends, explicitly selected, never auto-switched.** `memory` / `qdrant-local` / `bm25` are chosen once in the composition root. A missing index returns `200` with `status="unavailable"` instead of silently answering from a different retriever.

## Contents

- [Run the offline demo](#run-the-offline-demo-no-llm-provider-required)
- [Verified by CI](#verified-by-ci)
- [Overview](#overview)
- [Architecture](#architecture)
- [Authentication and Authorization](#authentication-and-authorization)
- [Audit Query API (read-only)](#audit-query-api-read-only)
- [PostgreSQL audit backend (opt-in)](#postgresql-audit-backend-opt-in)
- [MCP Interface (read-only)](#mcp-interface-read-only)
- [Knowledge-base Q&A (`/ask`)](#knowledge-base-qa-ask)
- [Key Features](#key-features)
- [Security Design](#security-design)
- [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

Security analysts spend much of their time turning raw telemetry into a defensible conclusion: is this indicator malicious, how serious is it, what should be done about it, and who approved that action. CyberSec Agent is a prototype that answers those questions over structured security events and threat-intelligence records — and that makes the answer auditable.

The design premise is that the language model is an untrusted component. The model contributes intent, tool selection, and prose. The risk level, the response plan, the approval requirement, and the audit trail are produced by deterministic rule engines that never read the model's output. The evaluation harness is built to check that this separation actually holds.

This is a research and engineering prototype. It is not production-ready and not an enterprise SOC product. It is not autonomous: it plans and requests approval, and it does not execute remediation actions against real infrastructure. All security data is synthetically generated by deterministic seed scripts.

## Architecture

```
                    HTTP client (curl / Swagger UI)
                                 │
                     FastAPI — app/api/main.py
       POST /chat    POST /ask    POST /triage    POST /resume    GET /audit/events
                                 │
              Authorization: Bearer <key> on all five; docs routes stay public
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
        /chat path                          /triage + /resume path
      SecurityAgent                            TriageService
      └─ ReAct graph                           └─ HITL graph, server-issued
         (agent ⇄ tools)                          thread handle, resume gate

        /ask path
        GroundedAskService
        └─ ReAct graph with exactly one read-only tool
           (search_knowledge_tool) + citation validator

  LangGraph StateGraph — app/core/graph.py
    agent ── LLM.bind_tools().ainvoke() ──►  ⚠ untrusted model output
      │
      ├─ has tool_calls ──► tools    name-routed execution; every failure becomes
      │         │                    a structured, retryable ToolMessage
      │         └────────► back to agent        (ReAct loop, capped iterations)
      │
      └─ no tool_calls ──► [ HITL graph only ]
                              │
                              plan           deterministic: collect evidence →
                              │              analyze_risk → plan_response
                              │              (does not read the message history)
                              │
                              policy_gate    deterministic: evaluate_policy(plan)
                              │              (does not read the message history)
                              │
                    ┌─────────┴─────────┐
                 allow               require_approval
                    │                   │
                   END            human_approval  ── interrupt() ⇒ graph pauses
                                                        │  resume with a decision
                                                        ▼

  Tools — app/tools/  (pure functions + thin LangChain wrappers)
    query_security_logs    structured event query over the seed log file
    query_threat_intel     exact-match IOC lookup (IP / domain / SHA256)
    analyze_risk           rule-based risk scoring          (pure function)
    plan_response          rule-based response planning     (pure function)

  Security layer — app/security/  (cross-cutting)
    policy.py           evaluate_policy(ResponsePlan) → allow | require_approval
    audit.py            plan digest + audit-record construction     (pure)
    auth.py             static API-key authentication → Principal(subject, role);
                        route-level role admission (viewer / analyst / approver)
    store_protocol.py   AuditStore — the ten-method contract that the graph,
                        the triage service and the API depend on
    store.py            SQLite backend (default): incidents / action_requests /
                        audit_logs / thread_owners — append-only, enforced by
                        eight database triggers
    store_postgres.py   PostgreSQL backend (explicit opt-in): same contract, same
                        append-only controls, twelve database triggers
                        read path: list_audit() — bounded, ordered SELECT only

  Data
    data/security_events.jsonl    144 synthetic events   (generated)
    data/threat_intel.jsonl        29 IOC records        (generated)
    data/audit.db                 created at runtime     (default SQLite backend)
    PostgreSQL                    opt-in; see "PostgreSQL audit backend" below
```

Two compiled graphs exist in one process: a plain ReAct graph for `/chat`, and a graph with the human-in-the-loop safety layer for `/triage` and `/resume`. They share one model client and one audit store.

### Why LangGraph

The orchestration layer is a `StateGraph` rather than a hand-written loop because the properties this prototype cares about are *control-flow* properties, and a graph makes them explicit and testable:

- **The safety path is structural.** `plan → policy_gate → human_approval` is wired into the graph, so the model cannot skip, reorder, or conditionally bypass it.
- **Tool execution is a named boundary.** The `agent` node's output is routed by tool name into a single `tools` node; model-produced code is never evaluated.
- **Pausing is a state, not a blocking call.** `interrupt()` suspends the graph at `human_approval`; a later request resumes it with a decision.
- **Control flow is testable offline.** The graph is compiled from plain functions, so nodes and edges are asserted with a scripted fake model and no network.

LangGraph supplies the state machine and the interrupt/resume mechanism — **not** the security properties above, which come from the graph's shape and from the policy gate being a node rather than a tool.

## Authentication and Authorization

Every business route (`/chat`, `/ask`, `/triage`, `/resume`, `/audit/events`) requires
an `Authorization: Bearer <key>` header. Authentication is **always on**: there is no
`AUTH_ENABLED=false` switch, and a missing or invalid `AUTH_API_KEYS` configuration
makes the process **refuse to start** rather than serve an open API.

Configuration holds **only SHA-256 digests** of the keys, never the keys themselves:

```bash
# 1. generate a key and store it now — it is shown only here
KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo "store this API key: $KEY"

# 2. put only its digest into AUTH_API_KEYS
printf '%s' "$KEY" | sha256sum
```

```dotenv
AUTH_API_KEYS=[{"sha256":"<64 lowercase hex>","subject":"alice","role":"analyst"},{"sha256":"<64 lowercase hex>","subject":"bob","role":"approver"},{"sha256":"<64 lowercase hex>","subject":"carol","role":"viewer"}]
```

`subject` is the **identity** (used for ownership and assignment); `role` is the
**authorization** (used for route admission). They are deliberately separate fields:
two people can hold the same `approver` role, which is what makes "the submitter and
the approver are different people" expressible. Subjects must be unique, and the digest
must be 64 lowercase hex characters.

### Role matrix

| Route | viewer | analyst | approver | Notes |
| --- | --- | --- | --- | --- |
| `POST /chat` | 403 | 200 | 200 | Plain LLM conversation. **Not** HITL / policy gated. |
| `POST /ask` | 403 | 200 | 200 | Knowledge-base Q&A with citations. **Not** HITL / policy gated. |
| `POST /triage` | 403 | 200 | 200 | Starts an HITL run; owner = authenticated subject. |
| `POST /resume` | 403 | 403 | 200 if assigned | Role admission **and** per-thread object authorization. |
| `GET /audit/events` | 200 (all) | 200 (own threads) | 200 (owned or assigned) | Scope rules below. |
| `GET /docs`, `/openapi.json`, `/redoc` | public | public | public | Interactive docs stay open by design. |

### Two layers of authorization

**Role admission** answers "does this authenticated principal have the right role for
this route" — a `viewer` calling `/triage` gets **403**.

**Object authorization** answers "may this principal act on *this specific thread*" —
an `approver` who was never assigned to a thread gets **404** from `/resume`,
byte-identical to the response for a thread that does not exist.

### Approver assignment and the self-approval prohibition

`POST /triage` requires an explicit `approvers` list. Before anything is written or
executed, the server checks that the list is non-empty, has no duplicates, contains no
unknown or non-`approver` subjects, and does **not** contain the caller — a submitter
can never approve their own run. Any violation returns **422** *before* any graph call
or ownership write. The owner is taken from the authenticated `subject`, never from the
request body.

### Error expectations

| Condition | Status | Body |
| --- | --- | --- |
| Missing / malformed / unknown credential | **401** | `{"detail":"authentication required"}` + `WWW-Authenticate: Bearer` |
| Authenticated, wrong role for the route | **403** | `{"detail":"insufficient role"}` |
| Authenticated approver not assigned to the thread | **404** | `{"detail":"not found"}` |
| Owner attempting to approve their own thread | **403** | `{"detail":"self-approval is prohibited"}` |
| analyst / approver reading audit without a `thread_id` | **403** | `{"detail":"an explicit thread scope is required for this role"}` |
| Invalid approver assignment | **422** | validation detail |

The 401 body is byte-identical for missing, malformed, and unknown credentials — the
response is deliberately **not** an oracle for "does this key exist". The 404 body is
likewise identical for "unrelated thread" and "unknown thread".

### Audit visibility

`GET /audit/events` is scope-filtered **before** any audit content is fetched — the
server never loads a thread the caller may not see:

- **viewer** — full read, including **historical threads that have no ownership row**
  (records written before per-thread ownership existed). `thread_id` is optional.
- **analyst** — must supply a `thread_id`, and may read only threads it **owns**.
- **approver** — must supply a `thread_id`, and may read only threads it **owns or was
  explicitly assigned to approve**.

An analyst / approver that omits `thread_id` gets **403**; one that supplies a thread
outside its scope (including an ownerless or unknown thread) gets **404**.

## Audit Query API (read-only)

The append-only audit trail is also readable over HTTP through a **read-only audit query endpoint**. It is an additional read interface over the existing authoritative audit store — not a second audit system and not a second audit writer. It is subject to the same authentication and per-thread scope rules as the rest of the API (see [Authentication and Authorization](#authentication-and-authorization)).

```bash
# a viewer reads the full trail; analyst / approver must add &thread_id=<id>
curl "http://127.0.0.1:8000/audit/events?limit=50&order=desc" \
  -H "Authorization: Bearer $VIEWER_KEY"
```

| Query parameter | Meaning |
| --- | --- |
| `thread_id` | exact-match filter on the graph execution thread |
| `event` | exact-match filter on one of the six closed audit event types |
| `limit` | bounded page size, 1–200 (default 50) |
| `order` | `desc` (default) or `asc` — deterministic ordering by `ts`, with a backend-provided tiebreaker (`rowid` on SQLite, `seq` on PostgreSQL) |

The response is a bare JSON array of `AuditRecord` records — there is no envelope. For an **authorized** scope, an empty result returns `200 []`. A `thread_id` outside the caller's scope does **not** return `200 []`: it returns `404` for analyst / approver, because scope is enforced before any content is fetched (see [Audit visibility](#audit-visibility)). A `viewer` may read any thread — including a historical thread with no ownership row — and receives `200 []` for a thread that has no events.

Boundary: the endpoint reads the **existing authoritative audit trail** through the configured `AuditStore`'s `list_audit` (parameterized `SELECT` only) — the same contract whether the backend is the default SQLite file or the opt-in PostgreSQL one. It does not mutate audit state and does not alter policy, approval, HITL, or checkpoint state. The client cannot choose the audit database path or connection string — the server keeps using the configured trusted store from settings — and no raw SQL is exposed.

This endpoint is **read-only** and **authenticated**: it requires a valid `Authorization: Bearer` credential, and the read scope is filtered to the caller's role before any audit content is fetched. Read-only is still not the same as tamper-proof — the audit trail is append-only by construction, not notarised (see [Known Limitations](#known-limitations)).

## PostgreSQL audit backend (opt-in)

SQLite remains the **default** audit backend. Setting no `AUDIT_*` variable gives the v0.1.0 behaviour exactly: a local SQLite file at `data/audit.db`.

PostgreSQL is an **explicit opt-in**. It is selected in one place — the application composition root — and nowhere else. No graph node, tool, or route branches on the backend; they all depend on the same `AuditStore` contract.

| Variable | Meaning |
| --- | --- |
| `AUDIT_BACKEND` | `sqlite` (default) or `postgres`. A closed set: an empty value fails startup validation. |
| `AUDIT_POSTGRES_DSN` | libpq connection string. Required when `AUDIT_BACKEND=postgres`. Read as a secret (`SecretStr`), so it never appears in a `repr`, a log line, or an error message. |
| `AUDIT_DB_PATH` | Optional; SQLite backend only. Defaults to `data/audit.db`. |
| `CHECKPOINT_DB_PATH` | Optional. Where the durable HITL checkpoints are stored. Defaults to `data/checkpoints.db`. Must **not** resolve to the same file as `AUDIT_DB_PATH` — the application refuses to start if it does, because the audit log and the checkpoint store are independent (see the persistence notes below). |

**No automatic fallback, and no dual-write.** If `AUDIT_BACKEND=postgres` and the DSN is missing or blank, the application refuses to start — it never quietly writes to SQLite instead. There is exactly one audit store per process, and nothing is written to two places.

**If PostgreSQL is unavailable.** The store connects lazily (constructing it performs no I/O), so an unreachable server surfaces at the first audit operation rather than at process start. From there the two paths behave differently — this is the current behaviour, not a design goal:

| Path | Current behaviour |
| --- | --- |
| Audit **write** (graph nodes, triage service) | The persistence failure propagates unhandled → HTTP **500**. |
| Audit **read** (`GET /audit/events`) | Mapped to a narrow internal type → HTTP **503**, body `{"detail": "audit store unavailable"}`. |

That asymmetry is a known limitation, not a guarantee; it is recorded under [Known Limitations](#known-limitations).

### Running the PostgreSQL integration tests

The repository ships an opt-in, isolated test service (`compose.postgres.yaml`) that shares no namespace, no volume, and no port with the default `compose.yaml` topology. `compose.yaml` itself is not modified.

```bash
# 1. start the disposable test server (loopback only, restart policy "no")
docker compose -f compose.postgres.yaml up -d --wait

# 2. provision the two separated roles
#    (a FRESH data directory runs this automatically; an existing volume does not)
docker exec -i cybersec-pg-test-postgres-1 psql -v ON_ERROR_STOP=1 \
  -U cybersec_test -d cybersec_test -f - < scripts/pg/bootstrap_roles.sql

# 3. apply the schema as the migration owner
CYBERSEC_PG_MIGRATION_DSN="postgresql://cybersec_migrator:cybersec_migrator_local_only@127.0.0.1:55432/cybersec_test" \
  uv run --no-sync alembic upgrade head

# 4. run the integration suite (explicitly — the default run deselects it)
uv run --no-sync pytest -m postgres -q

# 5. stop the service WITHOUT deleting the data volume
docker compose -f compose.postgres.yaml down
```

Two roles, deliberately separated:

| Role | Purpose | Privileges |
| --- | --- | --- |
| `cybersec_migrator` | migration owner; owns every schema object and runs Alembic | `CREATE` / `USAGE` on `public` |
| `cybersec_app` | application runtime; the only role the running app may use | `SELECT` / `INSERT` only — no `UPDATE`, `DELETE`, `TRUNCATE`, no DDL |

The credentials in `compose.postgres.yaml`, `scripts/pg/bootstrap_roles.sql`, and the CI workflow are **disposable local/CI test credentials**. They are not secrets, must never be reused, and must never front real data.

`docker compose -f compose.postgres.yaml down` stops the service and keeps the `pg-test-data` volume; only `down -v` deletes it. That is also the only supported way to reach a genuinely empty database, because the tables are append-only at the database level.

The offline suite does not collect these tests (`addopts = -m 'not postgres'` in `pyproject.toml`), so a machine without Docker still runs the whole default suite green. CI runs them in a separate job against a throwaway PostgreSQL service container.

### Scope and limits

- **Audit, incident, action-request and ownership data only.** The PostgreSQL backend stores the same four tables as the SQLite backend (`incidents`, `action_requests`, `audit_logs`, `thread_owners`). It does **not** store LangGraph checkpoints. Since v0.4.0 the checkpoints live in a separate local SQLite file (deliberately a different file from the audit database); that file is not part of this backend choice.
- **HITL state is no longer tied to one process lifetime — but it is single-writer.** Since v0.4.0 the checkpoints are stored in a local SQLite file, so a run paused at `human_approval` is not lost when the process that paused it exits. This is **not** a distributed store: it is single-writer, and a SQLite file on a shared volume is not safe for multiple replicas. Moving audit data to PostgreSQL does not change any of this.
- **Append-only is a database control, not tamper-proofing.** The triggers stop `UPDATE`, `DELETE`, and `TRUNCATE` issued through the application and through the runtime role. They do not stop a privileged administrator, a superuser, a table owner, or a forged `INSERT` of a plausible-looking record. The trail is append-only by construction, not notarised.
- **Not production-ready.** No high-availability or backup story, no operational guarantees, no rate limiting, no JWT/OIDC and no signed audit. This is an opt-in persistence backend for a prototype.
- **No performance or capacity claim.** The backend was exercised on one machine against one throwaway server. No throughput, latency, or scaling claim is made, and none should be inferred from it.
- **No exactly-once claim.** Audit writes are not deduplicated across processes; a retry can append a second record.
- **`/chat` is not the gated workflow.** `/chat` runs the ReAct graph only. The plan → policy gate → human-approval path exists on `/triage` and `/resume`. Switching the audit backend changes neither.

The design rationale and the exact boundary of these controls are in [docs/architecture.md](docs/architecture.md) §11.6.

## MCP Interface (read-only)

The same tool functions are also published over MCP (Model Context Protocol) as an **alternate read-only interface** for external clients. It does not replace the LangGraph agent path, the policy gate, or HITL.

```bash
python -m app.mcp.server        # local stdio server
```

Transport is **local stdio only**. No HTTP, SSE, or Streamable HTTP endpoint is implemented, and there is no MCP authentication server — the host process that spawns the server owns the identity boundary.

Three read-only tools are published:

| MCP tool | Behavior |
| --- | --- |
| `query_security_logs` | structured event query over the seed log file |
| `query_threat_intel` | exact-match IOC lookup (IP / domain / SHA256) |
| `analyze_risk` | rule-based risk scoring over already-collected evidence |

`plan_response` is **not** exposed through MCP: response planning remains behind the existing policy/HITL path and is not published as an MCP capability. MCP does not create a second authorization system.

Client-visible MCP schemas do not expose `data_path`, `logs_path`, or `intel_path`; the server binds repository-controlled data locations before delegating to the existing read-only domain functions. This is a property of the published interface as tested, not a general claim about filesystem access.

## Knowledge-base Q&A (`/ask`)

`POST /ask` is a fifth business route: it answers a natural-language question from the repository's version-controlled threat-intelligence corpus and returns **structural** groundedness plus validated citations. It is a read-only conversation path, like `/chat`, but with retrieval and a citation validator between the model and the client.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer $ALICE_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "Log4j 的 JNDI 查找为什么能被用来执行代码?"}'
```

The request body has exactly one field, `question`. There is deliberately no `top_k`, no `path` / `url` / `collection` / `index_path`, and no `corpus_version`: the client chooses *what to ask*, never *where to read*.

The response is a single object with five fields — the shape is frozen by
`tests/test_api/test_ask.py::test_response_has_exactly_the_frozen_fields`, and the
field semantics are declared in `app/api/schemas.py`:

| Field | Type | Meaning |
| --- | --- | --- |
| `answer` | `string` | The model's answer text. |
| `grounded` | `bool` | Structural verdict, not a correctness claim (see [below](#what-the-response-does-and-does-not-claim)). |
| `citations` | `array` | Valid citations only, each `{citation_id, source_id, source_type, title, url, score}`; ordered by `score` desc, then `citation_id` asc. `citation_id` **is** the document id. |
| `invalid_citations` | `array of string` | Citation ids that appear in the answer but were never retrieved in this request. |
| `retrieval` | `object` | `{status, hit_count, top_k}`; `status` ∈ `ok` / `empty` / `unavailable` / `not_used`. `top_k` is server-side and not client-controllable. |

No captured response body is shown here on purpose: producing one requires a live
provider, and pasting a scripted-model output would misrepresent it as a real answer.

### The retrieval backend is explicit, and never silently falls back

Three backends exist and are selected in the composition root only:

| `RAG_BACKEND` | Behaviour |
| --- | --- |
| `memory` (**default**) | Builds the in-memory hash-embedding index from the corpus at startup. Identical to the S2-C behaviour. |
| `qdrant-local` | Opens the Qdrant Local index at `RAG_QDRANT_PATH` (default `data/knowledge_index`). |
| `bm25` | Builds a **lexical** character 3-gram BM25 index from the validated corpus at startup. Nothing is persisted. |

`bm25` is a lexical ranker, **not** an embedding model. It scores documents by
character 3-gram overlap with IDF weighting (`k1 = 1.5`, `b = 0.75`); it has no
notion of synonymy or entailment. A question phrased with different words than
the corpus will not match, even if the answer is present — that is a property of
the method, not a bug to be tuned away. Because the index is built from the same
corpus snapshot that `load_corpus()` has already verified (manifest hash and
document count), there is no "index is older than the corpus" window and
therefore no six-field index identity to check. `bm25` deliberately has **no**
`verify()`: fabricating a check that can only pass would make "verified" a
meaningless word in the logs.

`retrieval.status = "empty"` means **this query shared no character 3-gram with
any document** — it does not mean the knowledge base has no answer. Under the
`memory` backend, `empty` is instead produced by the score threshold. In both
cases, only `invalid_citations` (not `empty`) is evidence that the corpus could
not support the answer.

There is no `auto` and no fallback. If `qdrant-local` is selected and the index is missing, stale, corrupt, or locked, the endpoint still returns **HTTP 200** with `retrieval.status = "unavailable"`, `grounded = false`, and `citations: []` — "no evidence this time" is a business result, not a server fault. Only a failed **model call** returns 502. Startup with a broken corpus (as opposed to a broken index) degrades `/ask` to the fixed 503 `{"detail":"rag service unavailable"}` and leaves `/chat`, `/triage`, `/resume`, and `/audit/events` unaffected. A `bm25` build failure is a corpus-level failure, so it takes the same 503 path; it never silently answers from the `memory` backend.

The index is an operational artifact, built by an explicit command — never at application startup:

```bash
python scripts/build_knowledge_index.py                  # build / rebuild
python scripts/build_knowledge_index.py --check          # verify against the corpus
```

### What the response does and does not claim

`grounded = true` is a **structural** statement: retrieval succeeded, at least one citation was produced, and no fabricated, unverifiable, or malformed citation appears. It is **not** a statement that the answer is factually correct, and it is **not** a statement that the retrieved documents are relevant — hash-based retrieval returns non-zero scores even for unrelated questions. Evidence is collected only from `type == "tool"` messages, so a model that writes a citation into its prose without calling the tool is reported as `invalid_citations`.

### Offline retrieval evaluation

The repository ships two labelled sets and a deterministic evaluator that calls **no LLM**:

| Set | Queries | Answerable | Role |
| --- | --- | --- | --- |
| `knowledge/eval/queries.jsonl` | 20 | 16 | **Frozen baseline** (S2). Never edited afterwards. Four gold documents share **zero** character 3-grams with their question. |
| `knowledge/eval/queries-s2f2-independent.jsonl` | 13 | 11 | **Supplementary validation set.** Labels were written before any ranking was inspected, but the set was created *after* the BM25 diagnosis and used to validate it — so it is **not** a fully blind test and must never be presented as an unused blind hold-out. Targets documents the first set never covers. |

```bash
python scripts/eval_rag.py --include-bm25
python scripts/eval_rag.py --include-bm25 \
    --queries knowledge/eval/queries-s2f2-independent.jsonl
```

It reports `Recall@5`, `MRR@5`, and `empty_rate` per backend. `memory` and `qdrant-local` return identical results by construction — the Qdrant Local adapter reuses the same `Retriever`, the same deterministic top-k, and the same citation ids.

Current numbers (descriptive, not a target to tune against):

| Set | `memory` (hash baseline) | `bm25` |
| --- | --- | --- |
| 20-query | 0.6562 / 0.6771 | 0.8125 / 0.9375 |
| independent 13 | 0.6818 / 0.7273 | 0.9545 / 0.9545 |

**The two sets must be reported separately and never merged or averaged.** The independent set has a median query↔document character-3-gram overlap of 0.500 against 0.278 for the 20-query set, and zero zero-overlap pairs against four — it is simply **easier**, so it can validate the *direction* of the lexical gain but not its magnitude, and it supports **no** generalisation claim. Neither number describes answer quality: `Recall@5` and `MRR@5` measure retrieval only, and the four zero-overlap gold documents are missed by **both** rankers because the lexical signal is absent. BM25 is a lexical ranker: it cannot recover a paraphrase that shares no character 3-gram with the corpus, so it does not solve semantic rewriting.

The full backend selection, failure semantics, and Qdrant Local's reconstruction constraints are documented in [docs/qdrant-local.md](docs/qdrant-local.md); the corpus provenance rules are in [knowledge/README.md](knowledge/README.md).

### Offline demo (no provider required)

The demo is introduced at the top of this README under [Run the offline demo](#run-the-offline-demo-no-llm-provider-required) — it is not restated here, so there is a single description to maintain.

Which outputs are real API behaviour, which are scripted, and which capabilities still require a live provider is spelled out in [docs/demo.md](docs/demo.md) — including the two Mermaid diagrams for citation validation and the HITL approval/resume sequence.

### Reproducible demo

```bash
# 1. build the Qdrant Local index (offline; no server, no Docker)
uv sync
uv run --no-sync python scripts/build_knowledge_index.py

# 2. start the API with the qdrant-local backend
RAG_BACKEND=qdrant-local uv run --no-sync uvicorn app.api.main:app --port 8000

# 3. ask an answerable question — expect 200, grounded=true, valid citations
curl -sS -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer $ALICE_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Log4j 的 JNDI 查找为什么能被用来执行代码?"}' | python -m json.tool

# 4. point the backend at a nonexistent index and ask again —
#    expect 200, grounded=false, citations=[], retrieval.status="unavailable"
#    (no silent fallback to the memory backend)
RAG_BACKEND=qdrant-local RAG_QDRANT_PATH=/tmp/no-such-index \
  uv run --no-sync uvicorn app.api.main:app --port 8001
curl -sS -X POST http://127.0.0.1:8001/ask \
  -H "Authorization: Bearer $ALICE_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Log4j 的 JNDI 查找为什么能被用来执行代码?"}' | python -m json.tool

# 5. ask an unanswerable question — the corpus has no support for it
#    (e.g. 蓝牙配对密钥协商缺陷); the endpoint reports no usable citation
#    rather than inventing one
curl -sS -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer $ALICE_KEY" -H "Content-Type: application/json" \
  -d '{"question": "蓝牙配对密钥协商缺陷"}' | python -m json.tool

# 6. offline retrieval evaluation over the frozen 20-query set
uv run --no-sync python scripts/eval_rag.py --qdrant-index-dir data/knowledge_index
```

The lexical backend needs no build step and no index directory:

```bash
# same API, different retriever — nothing is written to disk
RAG_BACKEND=bm25 uv run --no-sync uvicorn app.api.main:app --port 8000

curl -sS -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer $ALICE_KEY" -H "Content-Type: application/json" \
  -d '{"question": "日志组件为什么会在记录字符串时发起对外连接?"}' | python -m json.tool
# → 200, grounded=true when the model cites a real hit

curl -sS -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer $ALICE_KEY" -H "Content-Type: application/json" \
  -d '{"question": "zzzzz qqqqq"}' | python -m json.tool
# → 200, retrieval.status="empty", grounded=false, citations=[]
#   (no shared character 3-gram; this is not "the corpus has no answer")

# swap the backend on the running deployment only by restarting with another
# RAG_BACKEND value — there is no runtime switch and no auto-detection
```

**The two kinds of demo are not interchangeable.** Steps 2–5 above are the *real*
`/ask` path over HTTP: the API refuses to boot without `LLM_MODEL` / `LLM_API_KEY`,
and the `curl` calls reach a model, so they need a configured provider.
With no provider — or with an unreachable one — `/ask` returns **502**
`{"detail":"LLM service unavailable"}`; retrieval is not what failed.
Only step 1 (index build) and step 6 (`eval_rag.py`) are provider-free.

What can be exercised **without any provider** is narrower, and it is the part
that matters for retrieval quality:

- `scripts/eval_rag.py` calls no model at all and reports `Recall@5` / `MRR@5` /
  `empty_rate` over the labelled sets — this is the offline retrieval demo.
- `tests/test_api/test_ask.py` drives the same graph with a scripted model, so
  the endpoint's status, citation and RBAC semantics are covered offline.

Neither of those proves anything about answer text quality; see
[Known Limitations](#known-limitations).

`/ask` binds exactly one tool, `search_knowledge_tool`, and it takes no resource
locating parameter — no `path`, `url`, `collection`, or `index_path`. The model
chooses what to ask, not where to read.

## Key Features

- **LangGraph orchestration** — a five-node state graph (`agent`, `tools`, `plan`, `policy_gate`, `human_approval`) with a conditional ReAct loop and a business-level iteration cap separate from the framework's recursion limit.
- **Structured security-event analysis** — events are validated against a Pydantic model with closed vocabularies for event type, severity, and status, plus IPv4 and port validation. Malformed records fail at the boundary instead of flowing into the agent.
- **Security tool calling** — four declared tools registered in a single source of truth shared by the agent and the graph. The graph routes by tool name and never evaluates model-produced code.
- **Threat-intelligence lookup** — exact-match indicator lookup over IP, domain, and SHA256 records, deliberately not a similarity search: indicators are unique identifiers, and approximate matching would produce false associations.
- **Deterministic risk analysis** — a pure rule engine that scores evidence (intelligence hits, failed-login volume) with an explicit false-positive suppression path for indicators marked trusted.
- **Risk-aware response planning** — a pure rule engine that maps a risk assessment to a closed vocabulary of actions, each carrying its own approval requirement and reversibility flag. It consumes the assessment rather than re-deriving it.
- **Policy enforcement** — a rule-based policy gate implemented as a mandatory graph node rather than a tool, so the decision cannot be skipped.
- **Human-in-the-loop approval** — a genuine graph interrupt with a server-issued thread handle, a resume validation gate that rejects unknown, completed, expired, or lost threads, and a lazy approval-timeout lifecycle.
- **Append-only audit trail** — four tables (`incidents`, `action_requests`, `audit_logs`, `thread_owners`), eight anti-mutation triggers on SQLite (twelve on PostgreSQL, which adds `TRUNCATE` coverage), a closed six-event vocabulary that includes failure events, and approval state derived from the audit stream rather than stored as a mutable column.
- **Offline evaluation harness** — three baselines, five separately reported metric classes, an independent evidence oracle, metamorphic relations, and no composite score.
- **Real-provider evaluation harness** — a budget governor, a network-egress guard built on interpreter audit hooks, per-cell raw-record persistence with fsync, and frozen failure taxonomy.
- **Read-only MCP interface** — three read-only tools (security-log query, threat-intel lookup, deterministic risk analysis) published over a local stdio MCP server, reusing the existing tool functions as an alternate read-only surface. Response planning is not published, and no remote transport or authentication server is implemented.
- **Knowledge-base Q&A with validated citations** — `POST /ask` answers questions from a version-controlled, self-authored corpus through a single read-only retrieval tool, and returns a structural (not factual) groundedness verdict plus citations checked against the actual retrieved documents. The retrieval backend (`memory`, `qdrant-local`, or `bm25`) is explicit configuration selected in the composition root, with no silent fallback: a missing or corrupt index returns 200 + `unavailable`, a broken corpus or a failed `bm25` build returns 503, and neither ever degrades into an answer from a different retriever.
- **Fully offline retrieval evaluation** — a 20-query labelled set with `Recall@5` / `MRR@5` / `empty_rate` computed deterministically over both backends, with no LLM call and no network access.
- **Read-only audit query API** — `GET /audit/events` exposes the existing append-only audit trail as a bounded, deterministically ordered read surface (`thread_id` / `event` / `limit` / `order`). It is a read path only: no audit writes, no policy / approval / HITL / checkpoint mutation, and no client control over the audit database path. Reads are scope-filtered by role (viewer: all threads; analyst: owned threads; approver: owned or explicitly assigned threads), enforced before any content is fetched.
- **Optional PostgreSQL audit backend** — an explicit opt-in persistence backend behind the same `AuditStore` contract, with a separated migration-owner role and a runtime role limited to `SELECT` / `INSERT`. SQLite stays the default, there is no automatic fallback and no dual-write, and the PostgreSQL backend stores audit / incident / action-request data only — not LangGraph checkpoints.
- **Static API-key authentication** — every business route requires `Authorization: Bearer <key>`. Configuration stores only SHA-256 digests; authentication is always on and fail-closed (a missing or invalid key configuration stops the process at startup). Role admission (`viewer` / `analyst` / `approver`) is enforced per route as a FastAPI dependency.
- **Per-thread object authorization** — `POST /triage` records a single immutable ownership row (owner + full approver set) before the graph runs; `POST /resume` and `GET /audit/events` resolve the authenticated principal against that row before touching any graph, checkpointer, or audit content. Self-approval is rejected, and the audit `actor` is the authenticated principal, not a client-supplied string.
- **Fully offline test suite** — 2,283 tests that require no network access and no API key, plus a PostgreSQL integration suite (373 cases) that the default run deselects and a separate CI job runs against a throwaway server. All 373 were observed green at the released commit: GitHub Actions run [`40068176848`](https://github.com/luozhonglzw/cybersec-agent/actions/runs/40068176848) reported **373 passed, 0 skipped**. That is one CI run on one commit against one throwaway server — not a standing guarantee for later commits (see [Known Limitations](#known-limitations)).

## Security Design

**Untrusted LLM boundary.** The model produces intent and prose, never authority. It holds no shell, filesystem, or network capability, and its output is never the basis for an authorization decision. Concretely, the two nodes that produce authoritative security artifacts read only the caller-supplied indicator and the deterministic plan — they do not read the message history at all. Model output therefore cannot reach the risk level, the response plan, the approval requirement, or the audit record.

**Controlled tools.** Every capability the model can invoke is a pre-declared tool with a closed argument schema. Tool failures are converted into structured, retryable results rather than raised into the graph, and error payloads omit paths, arguments, and tracebacks. Response planning is deliberately withheld from the model on the approval path so that exactly one source of the plan exists.

**Deterministic planning and policy boundaries.** Risk scoring, response planning, and policy evaluation are pure functions: no I/O, no clock, no model call, no configuration input. The policy gate is a node, not a tool — a tool would be optional for the model to call, which would make the policy advisory instead of structural. Action properties such as whether an action requires approval, or whether it is reversible, are set by the rule engine.

**Human-in-the-loop.** High-impact actions pause the graph. Resuming requires a decision payload plus a server-issued thread handle, and the resume path verifies that the thread is genuinely awaiting approval before accepting anything. Whether an action needs approval is computed by the rule engine, never inferred by the model.

**Explicit approval assignment.** Each triage run records who owns it and which configured approvers may decide it. Only an assigned approver may resume that thread, and the submitter can never approve their own run — the assignment is a recorded fact, not a default derived from a role. Ownership is a single immutable row: owner and the full approver set are written together *before* the graph starts, so a partial assignment ("owner recorded, no approver") cannot exist.

**Ownership is registered before the graph runs.** If the graph then fails (for example a required data source is unavailable), the ownership row legitimately remains — it is immutable and is never silently deleted or reassigned. Such a thread is inert: no approval was requested, so it cannot be resumed; it is not readable by an unrelated analyst or approver; and re-registering the same `thread_id` fails loudly on the primary key. The only audit event written is a truthful `plan.failed`.

**Append-only audit trail.** Every decision and every approval is written to an append-only store, and history is enforced as history by the database: the store exposes no update or delete path, primary keys make duplicate writes fail loudly, and per-table triggers abort any UPDATE or DELETE issued outside the application (PostgreSQL adds a `TRUNCATE` trigger, because a statement-level wipe does not fire row-level triggers). There is no mutable status column — pending state is derived by querying for the absence of a terminal event. Failures are recorded alongside successes, and failure details deliberately exclude absolute paths and raw exception messages. The audit `actor` is the **authenticated principal**, not a client-supplied string: `POST /resume` ignores the request body's `operator` field and attributes the decision to the verified subject.

Known limitation: authentication is a **static API key**, and the audit `actor` field is authenticated but **not** non-repudiable — there is no signature and no separate record of a client-claimed identity, so a principal holding a valid key can still deny having acted. No non-repudiation claim is made anywhere in this prototype.

## Evaluation

Evaluation has two independent parts, and they answer different questions. Part A is deterministic, offline, and reproducible from the repository alone: it checks whether the architecture behaves as designed. Part B is a bounded, one-off observation with a real provider: it records what the system did under one specific configuration. Neither part is a security proof.

**(A) Offline architecture harness — deterministic, no network.** Three baselines — a rule-only pipeline, a ReAct graph with no policy gate, and the full agent with policy enforcement and HITL — are run over a fixed, hand-authored case set. Metrics are reported in five separate classes and are deliberately never combined into a single "agent score", because a weighted total would fold implementation self-consistency into correctness. The oracle recomputes evidence from raw records without importing the rule modules it is checking; case expectations carry provenance; and golden-set and dataset digests are emitted with every result so that it can be re-verified independently. Metamorphic relations are asserted without referencing any weight constant, and capabilities with no independent oracle are recorded as not-evaluable rather than scored zero.

**(B) Real-provider pilot — bounded, one configuration.** Evaluated the agent architecture with a controlled 108-run real-provider pilot and audited tool-use, injection-exposure, invocation, and architecture-invariant behavior.

Scope: 108 runs, 8 tasks, 4 parallel baseline configurations, n = 3 repetitions per cell, 197 logical LLM invocations; a single provider and endpoint configuration, requested model `deepseek-flash`, single execution window. The baselines are parallel experimental conditions, not a ranking. Token accounting is descriptive only; monetary cost is not evaluated; provider-side HTTP attempt counts are unobservable and recorded as unknown rather than zero.

Matched injection testing exposed model-side limitations: of 12 payload-bearing matched cases, 6 entered the model-visible context and were evaluable — all 6 followed the injected instruction, and none resisted. Cases that were never exposed are not counted as resistance. This is a single-provider, single-model, single-window observation over one matched injection task with a denominator of 6; it is not a claim about prompt-injection resistance in general, nor about the provider or the model family.

Architecture invariants — plan digest, policy outcome, and audit event sequence — were invariant across the pilot. They are regression guards on the separation between model behavior and the authoritative safety path, not a security score, and they hold by construction in the current design.

Pilot artifacts are anchored by artifact-level SHA256 digests; 34/34 data-quality checks passed; a secret scan over the final artifacts returned zero hits.

Nothing in this evaluation establishes that the system is secure, and no configuration is claimed to be the best.

## Known Limitations

This is a research and engineering prototype. The limitations below are properties of the current implementation, not a backlog of things that were overlooked.

**Not production-ready.** There is no production hardening: authentication is a static API key (no JWT/OIDC, no rotation, no expiry), there is no rate limiting, no signed audit, and no non-repudiation for the audit `actor` field. `GET /audit/events` is read-only *and* scope-filtered, but that is still not the same as tamper-proof or production-hardened.

**Historical threads may have no ownership row.** Audit records written before per-thread ownership existed carry no owner. A `viewer` can still read them; analyst and approver cannot resolve them, so a scoped audit read or a `/resume` against such a thread returns `404`.

**Single-service, single-instance persistence.** The application is one API process. By default it is backed by a local SQLite file; a PostgreSQL audit backend can be enabled explicitly, but it is optional and the deployment remains a single instance with no cache and no coordination between replicas. Nothing here is claimed to scale horizontally, and the Compose topology is deliberately one service with one named volume.

**Audit persistence is pluggable; checkpoint storage is a separate, local concern.** Choosing the PostgreSQL backend moves audit, incident, and action-request rows to PostgreSQL. It does **not** move LangGraph checkpoints: since v0.4.0 those live in a local SQLite file, deliberately a *different* file from the audit database (the application refuses to start if both point at the same path). A run paused at `human_approval` can therefore outlive the process that paused it, but this is a single-writer store, not a distributed one.

**Checkpoint storage is trusted storage.** Checkpoints are deserialised through a narrow allow-list — a fixed set of project models, `datetime`, and the LangGraph `Interrupt` type — and anything else is rejected with an error rather than reconstructed. That closes the "construct an arbitrary object from a type identifier stored in the database" path. It does **not** make a writable checkpoint database safe: whoever can write that file can still replace or corrupt a paused run's state. Treat that file as you treat the audit database.

**A recorded decision and a completed run are not atomic.** The audit store and the checkpoint store are independent and share no transaction, so a crash between "the decision was appended" and "the checkpoint advanced" is always possible. The decision is at most once by durable constraint; that is not the same claim as end-to-end exactly-once resume.

**Append-only is enforced by the database, but it is not tamper-proofing.** The anti-mutation triggers stop `UPDATE`, `DELETE`, and `TRUNCATE` through the application and through the runtime role. They do not stop a privileged administrator, a superuser, a table owner, or a forged `INSERT` that looks plausible. Pending-approval state is derived rather than stored, which removes one class of silent rewrite, but the trail is append-only by construction — not notarised.

**Audit write and audit read fail differently when the database is unavailable.** With the PostgreSQL backend, a failed audit **write** surfaces as HTTP 500, while a failed audit **read** (`GET /audit/events`) is mapped to HTTP 503. That asymmetry is the current behaviour, documented rather than promised.

**No exactly-once audit write.** Writes are not deduplicated across processes; a retry can append a second record. A repeated primary key fails loudly, but a retry that mints a new identifier does not.

**The MCP interface is local only.** It is a local stdio server with no remote transport and no authentication server; the identity boundary is the host process that spawns it.

**The agent path requires a provider.** `/chat`, `/triage`, and `/resume` run through the model, and there is no offline mode for that path. The deterministic and read-only parts of the system — tools, policy, audit, MCP — can be exercised without a provider; the agent cannot.

**`/ask` groundedness is structural, not factual.** `grounded = true` means the citations in the answer all point at documents actually retrieved in this request and no fabricated or malformed citation was found. It does not mean the answer is correct, and it does not mean the retrieved documents were relevant. The offline evaluator measures the retrieval layer only (`Recall@5` / `MRR@5` against the hash-embedding and BM25 baselines, on two separate labelled sets that must not be merged); answer-layer quality is not evaluated in this phase and must not be inferred from the retrieval numbers.

**Answer quality is NOT EVALUATED.** There is no measurement of answer correctness, faithfulness, completeness or hallucination rate for `/ask`. `Recall@5` / `MRR@5` describe the retrieval layer, and the two labelled sets (20 frozen + 13 supplementary) must always be reported separately: they are never merged, never averaged, and the supplementary set is **not** an unused blind hold-out — it was created after the BM25 diagnosis and is easier by construction. Retrieval scores are not evidence about answer text.

**The Qdrant Local index rebuild is not transactionally atomic.** Rebuilding writes points and index identity as two separate steps; a crash between them can leave a partially rebuilt collection. The mitigation is ordering, not atomicity: `create()` writes only a sentinel metadata record, the real six-field identity is committed last via `commit()` after an upsert/count self-check, so a crash anywhere inside the window leaves the sentinel in place and the index is rejected as `index-metadata-invalid` rather than silently serving stale vectors. That guarantees "identity visible last", not all-or-nothing. It also means a collection whose points are complete but whose identity was never committed is now treated as **failed** — rebuild it. See [docs/qdrant-local.md](docs/qdrant-local.md) §5.

**The PostgreSQL integration suite was verified in CI, not locally.** The 373 `postgres`-marked cases are deselected by default and run only in the dedicated CI job against a throwaway PostgreSQL 16 service container. At the released commit `9f93940`, GitHub Actions run [`40068176848`](https://github.com/luozhonglzw/cybersec-agent/actions/runs/40068176848) executed all 373 with **373 passed, 0 skipped** (the job fails if any case reports as skipped). They were **not** run against a local server for this release, because the development machine has no PostgreSQL instance. The evidence is therefore one CI run on one commit against one throwaway server: it is not a standing guarantee about later commits, and it says nothing about performance, capacity, or scale.

**Evaluation scope is narrow.** The real-provider pilot is 108 runs over 8 tasks, one provider and endpoint configuration, one requested model, one execution window, and a single matched injection task with a denominator of 6. It is a bounded observation, not a general result.

**Invariant checks are regression guards, not a security score.** Plan digest, policy outcome, and audit event sequence were invariant across the pilot; they hold by construction in the current design and are reported as architecture regression invariants, not as a measure of safety.

**Nothing here establishes that the system is secure.** No configuration is claimed to be best, and no claim is made about prompt-injection resistance in general or about any provider or model family.

## Quick Start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # create the environment and install dependencies

cp .env.example .env                       # then fill in the required variables (below)

uv run python scripts/seed_logs.py         # → data/security_events.jsonl  (144 events)
uv run python scripts/seed_threat_intel.py # → data/threat_intel.jsonl    (29 IOC records)

uv run uvicorn app.api.main:app --reload
```

The application validates its configuration at startup and **refuses to boot** if a
required value is missing or malformed. Two groups are required:

- **LLM access** — `LLM_MODEL` and `LLM_API_KEY`, both non-empty. `LLM_BASE_URL` is
  **optional**: leaving it unset falls back to the OpenAI SDK's default endpoint, so
  set it whenever your provider is not that default.
- **API authentication** — `AUTH_API_KEYS`. Without it the process stops with a
  validation error (`auth_api_keys: Field required`).

`data/audit.db` is created on first run. No `AUDIT_*` variable is needed for this
default path; to opt into PostgreSQL instead, see
[PostgreSQL audit backend (opt-in)](#postgresql-audit-backend-opt-in).

### 1. Generate API keys and configure `AUTH_API_KEYS`

Keys are generated by you and shown **once**; the configuration stores only their
SHA-256 digests, so a leaked config does not hand over a usable credential.

```bash
# repeat once per principal; keep each raw key in your secret store
KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo "alice's key (store it now): $KEY"
printf '%s' "$KEY" | sha256sum     # → put this digest into AUTH_API_KEYS
```

A minimal three-role configuration — **all values are placeholders**, replace the
digests with your own:

```dotenv
AUTH_API_KEYS=[{"sha256":"<alice-digest-64-hex>","subject":"alice","role":"analyst"},{"sha256":"<bob-digest-64-hex>","subject":"bob","role":"approver"},{"sha256":"<carol-digest-64-hex>","subject":"carol","role":"viewer"}]
```

Export the **raw** keys (not the digests) so the examples below can use them:

```bash
export ALICE_KEY="<alice raw key>"
export BOB_KEY="<bob raw key>"
export CAROL_KEY="<carol raw key>"
```

### 2. Call the API

Every business route requires the bearer header. Below, `alice` is an `analyst`, `bob`
an `approver`, and `carol` a `viewer`.

```bash
# Conversational analysis (analyst / approver)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ALICE_KEY" \
  -d '{"message": "分析一下 203.0.113.66 最近的登录活动"}'

# Knowledge-base Q&A over the shipped corpus (analyst / approver).
# This route calls the model: with no reachable provider it returns 502
# {"detail":"LLM service unavailable"} and retrieval never runs.
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ALICE_KEY" \
  -d '{"question": "Log4j 的 JNDI 查找为什么能被用来执行代码?"}'

# Start a triage run (analyst / approver). `approvers` is required and must list
# configured approver subjects other than the caller.
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ALICE_KEY" \
  -d '{"indicator": "203.0.113.66", "event_type": "login_failed", "approvers": ["bob"]}'

# Resume the paused run (approver only, and only an assigned one).
# thread_id comes from the /triage response.
curl -X POST http://127.0.0.1:8000/resume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $BOB_KEY" \
  -d '{"thread_id": "<from /triage>", "status": "approved", "operator": "bob"}'

# A viewer reads the whole trail
curl "http://127.0.0.1:8000/audit/events?limit=50&order=desc" \
  -H "Authorization: Bearer $CAROL_KEY"

# A scoped read — analyst / approver must supply thread_id
curl "http://127.0.0.1:8000/audit/events?thread_id=<from /triage>&order=asc" \
  -H "Authorization: Bearer $ALICE_KEY"
```

`operator` is retained for request compatibility only: the authoritative audit `actor`
is the authenticated principal, so `POST /resume` attributes the decision to `bob`
regardless of what the body says.

Expected errors (full table under
[Authentication and Authorization](#authentication-and-authorization)): no header or a
bad key → **401**; a `viewer` calling `/triage` → **403**; an unassigned approver
resuming a thread → **404**; a missing or invalid `approvers` list → **422**.

Interactive API documentation is served at <http://127.0.0.1:8000/docs> — the docs
routes themselves stay public.

Run the test suite — it needs no API key and makes no network calls:

```bash
uv run pytest -q
```

CI runs on GitHub-hosted Ubuntu in three independent jobs. The `test` job installs the locked `uv` environment and runs the offline pytest suite. The `test-postgres` job starts a throwaway PostgreSQL 16 service container, provisions the two separated roles, and runs `pytest -m postgres` explicitly — a missing or unreachable database fails that job rather than skipping it, and the job also fails if any case reports as skipped. The `docker` job builds the image and verifies the shipped corpus inside a real container: `load_corpus()` reads 40 documents / 80 chunks, `bm25` and `memory` retrieval each return hits whose `citation_id` resolves to that corpus, the default `RAG_BACKEND` is asserted to be `memory`, and the `/ask` route is registered. It performs **no** LLM call, so it is a packaging and retrieval check, not an end-to-end Q&A test.

All three jobs were observed green at the released commit: run [`40068176848`](https://github.com/luozhonglzw/cybersec-agent/actions/runs/40068176848) at `9f93940`, where `test` reported **2,283 passed, 373 deselected**, `test-postgres` reported **373 passed, 2,283 deselected** with 0 skipped, and `docker` succeeded. Each figure records one specific run on one specific commit — it is not a standing guarantee about the current head, and it is not a claim of universal Linux compatibility. Check the Actions tab for the latest result.

### Demo Walkthrough

There are two independent ways to see the system work, and they have different prerequisites.

**(A) Provider-backed agent walkthrough.** `/chat`, `/triage`, and `/resume` all run through the LangGraph agent, so all three require a configured provider (`LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` in `.env`); the application refuses to start without them. With valid credentials, a representative end-to-end run is:

```bash
uv run python scripts/seed_logs.py             # 1. generate the demo telemetry
uv run python scripts/seed_threat_intel.py     #    and the IOC records
uv run uvicorn app.api.main:app --reload       # 2. start the API

# 3. ask the agent a question — the model selects tools, the answer is prose
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ALICE_KEY" \
  -d '{"message": "帮我分析一下最近服务器有没有受到攻击"}'

# 4. start a triage run — a high-impact plan pauses for approval.
#    `approvers` must name a configured approver other than the caller.
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ALICE_KEY" \
  -d '{"indicator": "203.0.113.66", "event_type": "login_failed", "approvers": ["bob"]}'

# 5. resume the paused run — only an assigned approver may do this
#    (thread_id comes from /triage)
curl -X POST http://127.0.0.1:8000/resume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $BOB_KEY" \
  -d '{"thread_id": "<from /triage>", "status": "approved", "operator": "bob"}'

# 6. read the audit trail the run produced (viewer, or the assigned approver)
curl "http://127.0.0.1:8000/audit/events?thread_id=<from /triage>&limit=50&order=desc" \
  -H "Authorization: Bearer $BOB_KEY"
```

What is **stable** in this walkthrough: the four routes and their contracts, that `/triage` either ends in a policy decision or pauses for approval, that resuming requires a decision plus the server-issued handle, and that each step appends audit events. What is **not** stable: the model's exact wording, and whether a given indicator trips the approval gate — the risk level and the approval requirement are produced by the rule engine, not by the model.

**(B) Offline, provider-free capability demo.** The MCP interface is a separate, deterministic read-only surface over the same tool functions. It needs no provider credentials, because it never calls a model:

```bash
uv run python -m app.mcp.server     # local stdio MCP server
```

It publishes three read-only tools — `query_security_logs`, `query_threat_intel`, `analyze_risk` — over local stdio. This demonstrates the deterministic tool layer (structured event query, exact-match IOC lookup, rule-based risk scoring), and nothing else.

This is **not** an offline substitute for the agent path. It does not exercise the model, the ReAct loop, the plan stage, the policy gate, the human-approval interrupt, or the audit lifecycle — none of those are reachable without a provider.

### Running with Docker Compose

The repository also ships a minimal container deployment — a single `Dockerfile` and a `compose.yaml` at the repository root. It is an additional supported execution path, not a replacement for the local `uv` workflow above, and it reuses the same `.env` configuration.

```bash
cp .env.example .env          # same runtime configuration as the local workflow
docker compose up --build
```

The API is then served at <http://localhost:8000>, with interactive documentation at <http://localhost:8000/docs>.

**What is baked into the image.** The image installs the locked runtime dependencies, copies the application **and the knowledge corpus**, and generates the deterministic demo inputs at build time, so the container starts with its own data:

```
/app/knowledge/corpus.jsonl       40 self-authored documents  (version-controlled, byte-pinned)
/app/knowledge/manifest.json      corpus version + sha256 of the corpus
/app/data/security_events.jsonl   144 synthetic events  (image-baked, deterministic)
/app/data/threat_intel.jsonl       29 IOC records       (image-baked, deterministic)
```

`knowledge/` is copied because it is a **runtime input**: `app/rag/corpus.py` resolves the corpus relative to the installed package (`APPLICATION_ROOT / "knowledge"`), which is `/app/knowledge` in this image. Without it the container would boot normally and `/ask` would answer the fixed 503 for every backend, so it is part of the image contract rather than documentation. The same reasoning applies to the wheel, which ships `knowledge/` via Hatch `force-include`.

**What persists.** Compose declares exactly one service and one named volume:

| Item | Value |
| --- | --- |
| Service | `api` — the only service |
| Named volume | `audit-data` |
| Audit database path inside the container | `/data/audit.db` |
| Volume mount | `audit-data:/data` |
| Published port | `8000:8000` |

Mutable audit state is the only state that outlives a container. Because the volume mounts at `/data` and the demo inputs live under `/app/data`, the mount does **not** shadow the image-baked demo data. Recreating the container keeps the audit trail; removing the volume does not.

**Runtime configuration stays outside the image.** `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` are supplied at runtime from `.env` (or from the surrounding environment), exactly as in the local workflow. `.env` is excluded from the build context, so real credentials are never baked into the image — and, as with the local workflow, the application refuses to start without them.

**MCP is not a Compose service.** This deployment does not publish MCP over the network. It remains the local stdio server described under [MCP Interface (read-only)](#mcp-interface-read-only), started by whichever host process wants it; there is no remote MCP transport and no MCP authentication server.

**Scaling boundary.** The Compose deployment is intentionally a single API service with SQLite persistence — a local / demo / single-instance boundary. It is not a horizontally scaled or multi-replica design, and this topology deploys no external database or cache. The opt-in PostgreSQL backend is not part of it: the test service lives in a separate Compose project (`compose.postgres.yaml`), and enabling PostgreSQL for the application would require supplying a server that this topology does not provide.

**Compose version.** `compose.yaml` uses the optional `env_file` long syntax with `required: false`, which needs a Docker Compose release supporting that syntax (Compose ≥ 2.24).

## Project Structure

```
app/
  api/          FastAPI application, routes, request/response contracts
  core/         configuration, model client, LangGraph state graph, agent, triage service, /ask service
  schemas/      Pydantic models: events, threat intel, risk, response, approval, audit
  security/     policy engine, static API-key authentication and role admission, per-thread object authorization, append-only audit store (SQLite default, PostgreSQL opt-in)
  tools/        security tools (pure core functions + LangChain wrappers), including the read-only knowledge search tool
  rag/          retrieval layer: corpus loading and integrity checks, chunking, the hash-embedding baseline, the lexical BM25 ranker, citation construction (no network, no model)
  vectorstore/  Qdrant Local adapter (embedded mode, no server) with index identity and verified rebuild
  evaluation/   offline architecture harness and real-provider evaluation harness
  mcp/          read-only MCP adapters and the local stdio server
knowledge/      version-controlled, self-authored threat-intelligence corpus, its manifest, and the two labelled evaluation sets (a runtime input, not test data)
data/           generated seed data, the runtime audit database, and the Qdrant Local index (not committed)
docs/           architecture design notes and the Qdrant Local operating guide
scripts/        deterministic seed-data generators, the index builder, the offline retrieval evaluator; scripts/pg/ holds the role bootstrap SQL
tests/          2,283 offline tests, plus a PostgreSQL integration suite (deselected by default)
migrations/     Alembic environment and the hand-written baseline schema
alembic.ini     Alembic configuration — the connection URL comes from the environment, never from this file
Dockerfile      minimal container image (Phase 9.3-G); ships knowledge/ because the corpus is a runtime input
.dockerignore   build-context exclusions (secrets, host data, agent artifacts)
compose.yaml    single-service Compose topology with the audit-data volume
compose.postgres.yaml  opt-in, isolated PostgreSQL test service (outside the default topology)
```

## Roadmap

| Capability | Status |
| --- | --- |
| LangGraph orchestration | Implemented |
| Structured security-event analysis | Implemented |
| Security tool calling | Implemented |
| Threat-intelligence lookup | Implemented |
| Risk-aware response planning | Implemented |
| Policy enforcement and HITL | Implemented |
| Append-only audit trail | Implemented |
| Offline agent evaluation harness | Implemented |
| Real-provider evaluation pilot | Implemented |
| CI | Implemented |
| MCP interoperability | Implemented |
| Structured application logging | Implemented |
| Audit read / query API | Implemented |
| Static API-key authentication and role admission | Implemented |
| Per-thread object authorization and trusted audit actor | Implemented |
| PostgreSQL audit backend (opt-in) | Implemented |
| Containerized deployment | Implemented |
| Knowledge-base Q&A with validated citations (`/ask`) | Experimental (v0.5.0) |
| Pluggable retrieval backends: `memory` / `bm25` / `qdrant-local` | Experimental (v0.5.0) |
| Offline retrieval evaluation (`Recall@5` / `MRR@5`) | Experimental (v0.5.0) |

"Experimental" marks a capability that is implemented, tested and shipped, but whose quality claims are deliberately narrow: the retrieval numbers are offline lexical/embedding baselines on two small labelled sets, and answer-layer quality has not been evaluated at all. See [Known Limitations](#known-limitations).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.
