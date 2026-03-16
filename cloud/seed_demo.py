"""
Seed demo data for the playground.

Idempotent: safe to run multiple times. Creates a demo user
(playground@mengram.io) and populates it with realistic DevOps
agent memory — entities, facts, episodes, procedures, relations,
and embeddings.

Usage:
    railway run python -m cloud.seed_demo

Output: prints DEMO_USER_ID to set in Railway env vars.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database + embedder setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

from cloud.store import CloudStore       # noqa: E402
from cloud.embedder import CloudEmbedder  # noqa: E402

store = CloudStore(database_url=DATABASE_URL, pool_min=1, pool_max=2)
embedder = CloudEmbedder()

DEMO_EMAIL = "playground@mengram.io"

# ---------------------------------------------------------------------------
# Helper: get or create demo user
# ---------------------------------------------------------------------------

def get_or_create_user() -> str:
    """Return demo user_id, creating user if needed."""
    with store._cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email = %s", (DEMO_EMAIL,))
        row = cur.fetchone()
        if row:
            uid = str(row[0])
            log.info(f"Demo user already exists: {uid}")
            return uid
    uid = store.create_user(DEMO_EMAIL)
    log.info(f"Created demo user: {uid}")
    return uid


# ---------------------------------------------------------------------------
# Demo data definitions
# ---------------------------------------------------------------------------

ENTITIES = [
    {
        "name": "PostgreSQL",
        "type": "technology",
        "facts": [
            "PostgreSQL 16.2 running on Supabase (AWS us-east-1)",
            "Uses pgvector extension for 1536-dimensional embeddings",
            "Session-mode connection pooling on port 5432",
            "Daily automated backups at 03:00 UTC",
            "Primary database for all application data",
        ],
        "metadata": {"category": "database"},
    },
    {
        "name": "Railway",
        "type": "technology",
        "facts": [
            "Railway hosts the production API service",
            "Auto-deploys from the main branch on GitHub",
            "Currently running 1 replica with 1 gunicorn worker",
            "Uses health-check endpoint at /health for monitoring",
            "Environment variables managed via Railway dashboard",
        ],
        "metadata": {"category": "hosting"},
    },
    {
        "name": "Prometheus",
        "type": "technology",
        "facts": [
            "Prometheus collects metrics from the API service",
            "Scrape interval set to 15 seconds",
            "Alerting rules for error rate > 5% and p99 latency > 2s",
            "Retention period is 30 days",
        ],
        "metadata": {"category": "monitoring"},
    },
    {
        "name": "Grafana",
        "type": "technology",
        "facts": [
            "Grafana dashboards for API latency, error rates, and throughput",
            "Connected to Prometheus as primary data source",
            "Team has view-only access, only admins can edit dashboards",
            "PagerDuty integration for critical alerts",
        ],
        "metadata": {"category": "monitoring"},
    },
    {
        "name": "Redis",
        "type": "technology",
        "facts": [
            "Redis used for rate limiting and response caching",
            "Running Redis 7.2 on Railway",
            "Max memory set to 256MB with allkeys-lru eviction",
            "Cache TTL defaults to 5 minutes for search results",
        ],
        "metadata": {"category": "cache"},
    },
    {
        "name": "GitHub Actions",
        "type": "technology",
        "facts": [
            "CI/CD pipeline runs on every push to main",
            "Pipeline stages: lint, test, type-check, build",
            "Tests must pass before Railway auto-deploy triggers",
            "Average CI run takes about 3 minutes",
        ],
        "metadata": {"category": "ci-cd"},
    },
    {
        "name": "OpenAI API",
        "type": "technology",
        "facts": [
            "Uses gpt-4o-mini for text extraction and analysis",
            "text-embedding-3-large for vector embeddings at 1536 dimensions",
            "Monthly spend around $45 with current usage patterns",
            "Rate limit is 10,000 requests per minute on current tier",
        ],
        "metadata": {"category": "ai"},
    },
    {
        "name": "Docker",
        "type": "technology",
        "facts": [
            "Multi-stage Dockerfile for production builds",
            "Base image is python:3.11-slim",
            "Final image size is approximately 280MB",
            "docker compose used for local development with PostgreSQL and Redis",
        ],
        "metadata": {"category": "containerization"},
    },
    {
        "name": "Sarah Chen",
        "type": "person",
        "facts": [
            "Sarah Chen is the lead backend engineer",
            "She manages the database and deployment pipeline",
            "Prefers using Python for all backend services",
            "On-call rotation: Monday through Wednesday",
        ],
        "metadata": {"role": "engineer"},
    },
    {
        "name": "Incident #47",
        "type": "concept",
        "facts": [
            "Incident #47 was a production outage on March 3rd 2026",
            "Root cause: connection pool exhaustion under traffic spike",
            "Duration: 23 minutes of degraded service",
            "Resolution: increased pool_max from 2 to 5 and added connection timeout",
            "Post-mortem completed and shared with team on March 5th",
        ],
        "metadata": {"severity": "high"},
    },
]

RELATIONS = [
    ("PostgreSQL", "Supabase", "hosted_on", "PostgreSQL is hosted on Supabase"),
    ("Railway", "GitHub Actions", "triggered_by", "Railway deploys are triggered by GitHub Actions CI"),
    ("Prometheus", "Railway", "monitors", "Prometheus monitors the Railway-hosted API"),
    ("Grafana", "Prometheus", "visualizes", "Grafana visualizes Prometheus metrics"),
    ("Redis", "Railway", "hosted_on", "Redis runs on Railway"),
    ("Sarah Chen", "PostgreSQL", "manages", "Sarah manages the PostgreSQL database"),
    ("Sarah Chen", "Railway", "manages", "Sarah manages Railway deployments"),
    ("Incident #47", "PostgreSQL", "affected", "Incident #47 was caused by PostgreSQL pool exhaustion"),
    ("OpenAI API", "Redis", "cached_by", "OpenAI API responses are cached in Redis"),
    ("Docker", "Railway", "deployed_to", "Docker images are deployed to Railway"),
]

EPISODES = [
    {
        "summary": "Database connection pool exhaustion caused 23-minute outage",
        "context": "Traffic spike from Hacker News post overwhelmed the connection pool. pool_max was set to 2, which couldn't handle concurrent requests. Synchronous psycopg2 blocked the async event loop, causing cascading failures.",
        "outcome": "Increased pool_max from 2 to 5, added 30-second connection timeout, set up connection pool monitoring in Grafana. Service restored after 23 minutes.",
        "participants": ["Sarah Chen", "PostgreSQL", "Incident #47"],
        "emotional_valence": "negative",
        "importance": 0.9,
        "happened_at": "2026-03-03",
    },
    {
        "summary": "Migrated embedding model from text-embedding-ada-002 to text-embedding-3-large",
        "context": "Ada embeddings were 1536-dim but lower quality. The new text-embedding-3-large model provides better semantic accuracy at the same dimension. Required re-embedding all existing data (~50K vectors).",
        "outcome": "Migration completed successfully over a weekend. Search relevance improved by approximately 15% based on manual evaluation. Total re-embedding cost was $12.",
        "participants": ["OpenAI API", "PostgreSQL"],
        "emotional_valence": "positive",
        "importance": 0.7,
        "happened_at": "2026-02-15",
    },
    {
        "summary": "Set up PagerDuty integration with Grafana for on-call alerting",
        "context": "Previously alerts only went to a Slack channel that nobody watched at night. After Incident #47, the team decided to add PagerDuty for critical alerts. Configured escalation policy: primary on-call gets paged first, then the whole team after 10 minutes.",
        "outcome": "PagerDuty integration working. First real alert triggered 3 days later for a spike in 5xx errors — on-call engineer responded in 4 minutes.",
        "participants": ["Grafana", "Sarah Chen"],
        "emotional_valence": "positive",
        "importance": 0.6,
        "happened_at": "2026-03-07",
    },
    {
        "summary": "Redis out-of-memory crash during Hacker News traffic spike",
        "context": "HN front page drove 10x normal traffic. Redis maxmemory was 128MB with no eviction policy set, causing OOM crash. All cached search results were lost, putting full load on PostgreSQL.",
        "outcome": "Increased Redis maxmemory to 256MB, set allkeys-lru eviction policy, added memory usage alert at 80%. Recovery took 8 minutes after config change.",
        "participants": ["Redis", "PostgreSQL"],
        "emotional_valence": "negative",
        "importance": 0.8,
        "happened_at": "2026-02-28",
    },
    {
        "summary": "Quarterly infrastructure cost review and optimization",
        "context": "Reviewed all cloud spending: Railway $25/mo, Supabase $25/mo, OpenAI API $45/mo, domain $12/yr. Total burn rate about $96/month. Identified that 30% of OpenAI embedding calls were duplicates that could be cached.",
        "outcome": "Added embedding cache in Redis, reducing OpenAI costs by approximately 25%. New monthly burn rate: ~$84/month. Next review scheduled for June.",
        "participants": ["Sarah Chen", "OpenAI API", "Railway"],
        "emotional_valence": "neutral",
        "importance": 0.5,
        "happened_at": "2026-03-10",
    },
]

PROCEDURES = [
    {
        "name": "Deploy to Production",
        "trigger_condition": "When deploying a new version to production",
        "steps": [
            {"step": 1, "action": "Run the full test suite locally: pytest --tb=short"},
            {"step": 2, "action": "Check CI status on GitHub Actions — all checks must be green"},
            {"step": 3, "action": "Review the diff since last deploy: git log --oneline main..HEAD"},
            {"step": 4, "action": "If any database migrations, run them on staging first and verify"},
            {"step": 5, "action": "Check if any ALTER TABLE on tables with > 1M rows — if yes, use --lock-timeout=5s and schedule during low traffic (02:00-04:00 UTC)"},
            {"step": 6, "action": "Merge PR to main — Railway auto-deploys within 2 minutes"},
            {"step": 7, "action": "Monitor Grafana error rate dashboard for 15 minutes post-deploy"},
        ],
        "entity_names": ["Railway", "GitHub Actions", "PostgreSQL", "Grafana"],
        "success_count": 8,
        "fail_count": 2,
    },
    {
        "name": "Handle Database Incident",
        "trigger_condition": "When database errors or connection issues are detected",
        "steps": [
            {"step": 1, "action": "Check Grafana dashboard for connection pool usage and error rates"},
            {"step": 2, "action": "Run: SELECT count(*) FROM pg_stat_activity WHERE state = 'active' to check active connections"},
            {"step": 3, "action": "If pool exhaustion: increase pool_max in Railway env vars and restart"},
            {"step": 4, "action": "If slow queries: check pg_stat_statements for queries > 5s and add missing indexes"},
            {"step": 5, "action": "Verify recovery by checking /health endpoint returns 200"},
            {"step": 6, "action": "Write post-mortem within 48 hours and share with team"},
        ],
        "entity_names": ["PostgreSQL", "Grafana", "Sarah Chen"],
        "success_count": 3,
        "fail_count": 1,
    },
    {
        "name": "Rotate API Keys",
        "trigger_condition": "When API keys need rotation (quarterly or after security incident)",
        "steps": [
            {"step": 1, "action": "Generate new API key in the provider dashboard (OpenAI, Cohere, etc.)"},
            {"step": 2, "action": "Update the key in Railway environment variables — do NOT commit to git"},
            {"step": 3, "action": "Trigger a Railway restart to pick up the new env var"},
            {"step": 4, "action": "Verify the service is working with the new key by calling /health and running a test search"},
            {"step": 5, "action": "Revoke the old key in the provider dashboard"},
            {"step": 6, "action": "Update the team password manager with the new key prefix"},
        ],
        "entity_names": ["OpenAI API", "Railway"],
        "success_count": 4,
        "fail_count": 0,
    },
]


# ---------------------------------------------------------------------------
# Seeding functions
# ---------------------------------------------------------------------------

def clear_demo_data(user_id: str):
    """Remove all existing demo data for clean re-seed."""
    with store._cursor() as cur:
        # Entities cascade-deletes facts, embeddings, relations, knowledge
        cur.execute("DELETE FROM entities WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM episodes WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM procedures WHERE user_id = %s", (user_id,))
    log.info("Cleared existing demo data")


def seed_entities(user_id: str) -> dict[str, str]:
    """Create entities + facts, return {name: entity_id} map."""
    name_to_id = {}
    for ent in ENTITIES:
        eid = store.save_entity(
            user_id=user_id,
            name=ent["name"],
            type=ent["type"],
            facts=ent.get("facts", []),
            metadata=ent.get("metadata", {}),
        )
        name_to_id[ent["name"]] = eid
        log.info(f"  Entity: {ent['name']} → {eid}")
    return name_to_id


def seed_entity_embeddings(name_to_id: dict[str, str]):
    """Generate and save embeddings for each entity's facts."""
    for ent in ENTITIES:
        eid = name_to_id[ent["name"]]
        # Build chunk: entity name + all facts
        chunk = f"{ent['name']}: " + ". ".join(ent.get("facts", []))
        emb = embedder.embed(chunk)
        store.save_embedding(eid, chunk, emb)
    log.info(f"  Entity embeddings: {len(name_to_id)}")


def seed_relations(user_id: str, name_to_id: dict[str, str]):
    """Create relations between entities."""
    created = 0
    for source_name, target_name, rel_type, description in RELATIONS:
        source_id = name_to_id.get(source_name)
        if not source_id:
            log.warning(f"  Skipping relation: source '{source_name}' not found")
            continue
        # _save_relation auto-creates the target entity if it doesn't exist
        store._save_relation(
            user_id=user_id,
            source_entity_id=source_id,
            source_name=source_name,
            rel={
                "target": target_name,
                "type": rel_type,
                "description": description,
                "direction": "outgoing",
            },
        )
        created += 1
    log.info(f"  Relations: {created}")


def seed_episodes(user_id: str) -> list[str]:
    """Create episodes, return list of episode IDs."""
    episode_ids = []
    for ep in EPISODES:
        eid = store.save_episode(
            user_id=user_id,
            summary=ep["summary"],
            context=ep.get("context"),
            outcome=ep.get("outcome"),
            participants=ep.get("participants", []),
            emotional_valence=ep.get("emotional_valence", "neutral"),
            importance=ep.get("importance", 0.5),
            happened_at=ep.get("happened_at"),
        )
        episode_ids.append(eid)
        log.info(f"  Episode: {ep['summary'][:60]}... → {eid}")
    return episode_ids


def seed_episode_embeddings(episode_ids: list[str]):
    """Generate and save embeddings for episodes."""
    for eid, ep in zip(episode_ids, EPISODES):
        chunk = f"{ep['summary']}. {ep.get('context', '')}. {ep.get('outcome', '')}"
        emb = embedder.embed(chunk)
        store.save_episode_embedding(eid, chunk, emb)
    log.info(f"  Episode embeddings: {len(episode_ids)}")


def seed_procedures(user_id: str) -> list[str]:
    """Create procedures, return list of procedure IDs."""
    proc_ids = []
    for proc in PROCEDURES:
        pid = store.save_procedure(
            user_id=user_id,
            name=proc["name"],
            trigger_condition=proc.get("trigger_condition"),
            steps=proc.get("steps", []),
            entity_names=proc.get("entity_names", []),
        )
        # Update success/fail counts directly
        with store._cursor() as cur:
            cur.execute(
                """UPDATE procedures SET success_count = %s, fail_count = %s
                   WHERE id = %s""",
                (proc.get("success_count", 0), proc.get("fail_count", 0), pid),
            )
        proc_ids.append(pid)
        log.info(f"  Procedure: {proc['name']} → {pid}")
    return proc_ids


def seed_procedure_embeddings(proc_ids: list[str]):
    """Generate and save embeddings for procedures."""
    for pid, proc in zip(proc_ids, PROCEDURES):
        steps_text = " ".join(s["action"] for s in proc.get("steps", []))
        chunk = f"{proc['name']}: {proc.get('trigger_condition', '')}. Steps: {steps_text}"
        emb = embedder.embed(chunk)
        store.save_procedure_embedding(pid, chunk, emb)
    log.info(f"  Procedure embeddings: {len(proc_ids)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Seeding demo data for playground ===")

    user_id = get_or_create_user()

    log.info("Clearing existing demo data...")
    clear_demo_data(user_id)

    log.info("Seeding entities + facts...")
    name_to_id = seed_entities(user_id)

    log.info("Generating entity embeddings...")
    seed_entity_embeddings(name_to_id)

    log.info("Seeding relations...")
    seed_relations(user_id, name_to_id)

    log.info("Seeding episodes...")
    episode_ids = seed_episodes(user_id)

    log.info("Generating episode embeddings...")
    seed_episode_embeddings(episode_ids)

    log.info("Seeding procedures...")
    proc_ids = seed_procedures(user_id)

    log.info("Generating procedure embeddings...")
    seed_procedure_embeddings(proc_ids)

    log.info("=== Done ===")
    log.info(f"Entities: {len(ENTITIES)}, Episodes: {len(EPISODES)}, Procedures: {len(PROCEDURES)}, Relations: {len(RELATIONS)}")
    print(f"\nDEMO_USER_ID={user_id}")
    print("Set this in Railway environment variables.")


if __name__ == "__main__":
    main()
