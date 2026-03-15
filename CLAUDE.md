# CLAUDE.md — Mengram AI Memory Platform

This file provides guidance for AI assistants (Claude Code and similar) working in this codebase.

---

## Project Overview

**Mengram** is an open-source AI memory platform that gives LLMs and AI agents persistent, evolving memory. It supports three memory types:

- **Semantic Memory** — Facts, preferences, relationships
- **Episodic Memory** — Events, decisions, outcomes with timeline
- **Procedural Memory** — Workflows that evolve from failures via LLM analysis

It is a free alternative to Mem0 with two deployment modes: **local** (SQLite + Obsidian vault) and **cloud** (PostgreSQL + pgvector + FastAPI).

**PyPI**: `mengram-ai` v2.18.0 | **npm**: `mengram-ai` v2.16.0 | **Python**: 3.10+

---

## Repository Structure

```
mengram/
├── mengram.py                  # Main SDK entry point (Memory + Mengram factory)
├── mengram_middleware.py        # AutoMemory: auto-recall/remember wrapper
├── importer.py                  # ChatGPT / Obsidian / file importers
├── cli.py                       # CLI: init, server, status, hook, import, search
├── __init__.py                  # Package exports

├── engine/                      # LOCAL MODE (self-hosted, offline)
│   ├── brain.py                 # MengramBrain: remember() + recall() orchestrator
│   ├── extractor/
│   │   ├── conversation_extractor.py  # LLM-powered knowledge extraction
│   │   └── llm_client.py              # ABC + Anthropic/OpenAI/Ollama/Mock clients
│   ├── vault_manager/
│   │   └── vault_manager.py     # .md file CRUD (Obsidian vault)
│   ├── graph/
│   │   └── knowledge_graph.py   # SQLite knowledge graph (entities + relations)
│   ├── vector/
│   │   ├── embedder.py          # sentence-transformers (local, 80MB model)
│   │   └── vector_store.py      # SQLite vector store + HNSW search
│   ├── parser/
│   │   └── markdown_parser.py   # Vault .md parsing (frontmatter, wikilinks, tags)
│   └── retrieval/
│       └── hybrid_search.py     # Vector + knowledge graph combined retrieval

├── cloud/                       # CLOUD MODE (production, mengram.io)
│   ├── api.py                   # FastAPI server (~6200 lines)
│   ├── store.py                 # PostgreSQL + pgvector backend (~5400 lines)
│   ├── client.py                # CloudMemory sync SDK
│   ├── async_client.py          # AsyncMengramClient (httpx)
│   ├── evolution.py             # Procedure evolution engine
│   ├── embedder.py              # OpenAI embeddings API client
│   ├── docker-compose.yml       # Local cloud dev (Postgres + Redis)
│   └── __init__.py

├── api/                         # Integration servers
│   ├── mcp_server.py            # MCP server for local vault
│   ├── cloud_mcp_server.py      # MCP server for cloud API (~1350 lines)
│   └── rest_server.py           # REST API wrapper

├── integrations/
│   ├── langchain.py             # LangChain chat history + retriever tools
│   ├── crewai.py                # CrewAI tools (5 memory operations)
│   └── openclaw/                # OpenClaw skill plugin

├── sdk/
│   ├── js/                      # JavaScript SDK (npm mengram-ai)
│   │   ├── index.js
│   │   └── index.d.ts           # TypeScript declarations
│   └── langchain-mengram/       # pip langchain-mengram plugin

├── examples/
│   ├── devops-agent/            # CloudMemory + procedure evolution demo
│   ├── customer-support-agent/  # CrewAI integration demo
│   ├── personal-assistant/      # LangChain cognitive profile demo
│   └── n8n/                     # n8n workflow integration

├── obsidian-plugin/             # Obsidian vault plugin (TypeScript)
├── vscode-mengram/              # VSCode extension (TypeScript)
├── tests/
│   ├── test_parser.py           # Markdown parser tests
│   └── test_importer.py         # Importer tests (ChatGPT, Obsidian, rate limiting)
├── benchmarks/                  # LOCOMO benchmark scripts
├── blog/                        # Blog posts as Markdown

├── pyproject.toml               # Python package config + pytest settings
├── requirements.txt             # Full dependency list
├── Dockerfile.selfhost          # Self-hosted API image (python:3.12-slim)
├── docker-compose.yml           # Dev env: PostgreSQL + Redis + Mengram API
├── railway.json                 # Railway.app deployment config
├── config.example.yaml          # Configuration template
├── README.md                    # Main user documentation (440 lines)
└── ARCHITECTURE.md              # System architecture documentation
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10–3.13 |
| Web Framework | FastAPI + Uvicorn + Gunicorn |
| Cloud DB | PostgreSQL 16 + pgvector (HNSW indexing) |
| Local DB | SQLite (vector store + knowledge graph) |
| Search | HNSW + BM25 hybrid + LLM re-ranking |
| LLM Providers | Anthropic Claude, OpenAI GPT, Ollama |
| Local Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Cloud Embeddings | OpenAI Embeddings API |
| Cache | Redis |
| Async HTTP | httpx |
| Email | Resend |
| JS SDK | Plain JS + TypeScript types |

---

## Development Setup

### Install for development
```bash
pip install -e .           # minimal
pip install -e ".[all]"    # all optional features
pip install -e ".[api,async,langchain,crewai]"  # specific features
```

### Configuration
```bash
cp config.example.yaml ~/.mengram/config.yaml
# Edit: vault_path, LLM provider/key, embeddings model
```

### Run tests
```bash
pytest tests/
pytest tests/test_parser.py -v
pytest tests/test_importer.py -v
```

### CLI usage
```bash
mengram init          # interactive setup wizard
mengram server        # start MCP server for Claude Desktop
mengram status        # check configuration
mengram hook install  # install Claude Code session hook
mengram search "query"
mengram profile
```

### Local cloud dev environment
```bash
docker-compose up     # PostgreSQL 16 + Redis + Mengram API on :8420
```

### Self-hosted deployment
```bash
docker build -f Dockerfile.selfhost -t mengram:latest .
docker run -e DATABASE_URL=postgresql://... -p 8420:8420 mengram:latest
```

---

## Key Conventions

### Python Style
- **Classes**: `PascalCase` (`MengramBrain`, `CloudMemory`, `VaultManager`)
- **Functions/methods**: `snake_case` (`remember_text`, `recall`, `search_all`)
- **Constants**: `UPPER_SNAKE_CASE` (`DEFAULT_HOME`, `EXTRACTION_PROMPT`)
- **Type hints**: Required; use Python 3.10+ union syntax (`list[dict]`, `str | None`)
- **Docstrings**: Present on public classes and methods

### Architecture Patterns
- **Two-mode design**: Local and Cloud expose the same public API surface
- **Factory pattern**: `Mengram()` returns the right client based on config
- **Abstract base classes**: `LLMClient` ABC for pluggable LLM backends
- **Lazy properties**: `@property` with `_cache` pattern (graph, vector_store)
- **Dataclasses**: Use for structured data (`MemoryItem`, `SearchResult`, `Entity`)
- **Async variants**: Separate `AsyncMengram` / `AsyncMengramClient` classes (not mixed)
- **Multi-user scoping**: Thread `user_id` through all public methods; default is `"default"`

### Optional Dependencies
Guard all optional imports with try/except:
```python
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
```

### Error Handling
- Custom exceptions preferred (e.g., `QuotaExceededError`)
- Validate only at system boundaries (user input, external APIs)
- Avoid overly defensive internal checks

---

## Core Concepts

### Memory Types

| Type | Storage (Local) | Storage (Cloud) | Key Method |
|------|-----------------|-----------------|------------|
| Semantic | `.md` files + SQLite graph | `entities` + `pgvector` table | `add()` / `search()` |
| Episodic | `.episodes.json` per entity | `episodes` table | `episodes()` |
| Procedural | `.procedures.json` per entity | `procedures` table | `procedures()` + `procedure_feedback()` |

### Extraction Flow
```
Conversation text
       ↓
ConversationExtractor (LLM prompt)
       ↓
ExtractionResult {entities, facts, relations, episodes, procedures}
       ↓
VaultManager (.md) / CloudStore (PostgreSQL)
       ↓
Embedder + KnowledgeGraph (sync)
```

### Procedure Evolution
Procedures improve automatically from failures:
```
v1: build → push → deploy
        ↓ FAILURE (migration missing)
v2: build → run_migrations → push → deploy
        ↓ FAILURE (memory check missing)
v3: build → run_migrations → check_memory → push → deploy ✅
```
Triggered via `procedure_feedback(procedure_id, success=False, context="...")`.

### Hybrid Search
All recall combines:
1. Vector similarity (HNSW on embeddings)
2. Knowledge graph traversal (2-hop BFS)
3. BM25 keyword matching (cloud only)
4. LLM re-ranking (cloud, top-N results)

### Multi-User Isolation
- **API key** → account-level auth (billing, rate limits)
- **`user_id`** → data scoping within an account
- Same API key can serve multiple end-users by varying `user_id`

---

## Important Files for Common Tasks

| Task | Files |
|------|-------|
| Change extraction logic | `engine/extractor/conversation_extractor.py` |
| Add LLM provider | `engine/extractor/llm_client.py` |
| Modify vault storage | `engine/vault_manager/vault_manager.py` |
| Change cloud API endpoints | `cloud/api.py` |
| Change cloud DB schema | `cloud/store.py` |
| Add CLI command | `cli.py` |
| Add LangChain integration | `integrations/langchain.py` |
| Add CrewAI tool | `integrations/crewai.py` |
| Add MCP tool | `api/cloud_mcp_server.py` |
| Update JS SDK | `sdk/js/index.js` + `sdk/js/index.d.ts` |
| Add example | `examples/<name>/main.py` |

---

## Testing

Tests use **pytest** with no extra plugins. Test configuration is in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- `tests/test_parser.py` — unit tests for markdown parsing, graph connections
- `tests/test_importer.py` — unit tests for ChatGPT/Obsidian import, rate limiting

When adding features, add or update tests in `tests/`. Use `MockLLMClient` from `engine/extractor/llm_client.py` for tests that would otherwise call a real LLM.

---

## Deployment

### Environment Variables (cloud mode)
```
DATABASE_URL          postgresql://user:pass@host:5432/db
REDIS_URL             redis://host:6379
OPENAI_API_KEY        sk-...
ANTHROPIC_API_KEY     sk-ant-...
RESEND_API_KEY        re_...
SECRET_KEY            (JWT signing)
```

### Ports
| Service | Port |
|---------|------|
| Mengram API | 8420 |
| PostgreSQL | 5432 |
| Redis | 6379 |
| MCP server (local) | stdio |

---

## What NOT to Do

- **Do not mix sync and async code** — `AsyncMengram` and `Mengram` are separate; keep them separate.
- **Do not add hard dependencies for optional features** — use try/except guards and `extras_require` in `pyproject.toml`.
- **Do not bypass multi-user scoping** — always pass `user_id` through; never query without it in cloud mode.
- **Do not edit `cloud/api.py` and `cloud/store.py` lightly** — they are ~6200 and ~5400 lines respectively and touch production data. Understand the existing patterns before modifying.
- **Do not commit secrets** — `config.yaml`, `.env`, API keys; all are in `.gitignore`.
- **Do not add validation inside internal functions** — validate only at API/CLI boundaries.
