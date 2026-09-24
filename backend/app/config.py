"""Central config: load environment once, expose as simple module-level constants."""
import os

import yaml
from dotenv import load_dotenv

# Load backend/.env (next to this app package's parent dir). Safe if the file is missing.
load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
SAMPLE_BOT_URL: str = os.getenv("SAMPLE_BOT_URL", "http://localhost:8001/chat")
# LLM-backed RAG agent (real, non-deterministic test target) on :8002.
SAMPLE_RAG_BOT_URL: str = os.getenv("SAMPLE_RAG_BOT_URL", "http://localhost:8002/chat")

# PostgreSQL connection string. Override in .env to point at another host/db.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://localhost:5432/agentshield"
)

# Demo safety: cap scenarios per run so a live run finishes well under ~90s.
MAX_SCENARIOS: int = int(os.getenv("MAX_SCENARIOS", "10"))

# --- Temporal ----------------------------------------------------------------
# Where the Temporal server is. `temporal server start-dev` listens on 7233 by default
# and serves its Web UI on 8233. Nothing in the app requires Temporal yet — these only
# take effect for the worker and for code that explicitly asks for a client.
TEMPORAL_ADDRESS: str = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_NAMESPACE: str = os.getenv("TEMPORAL_NAMESPACE", "default")
# The queue a worker polls and a client targets. One queue for now; splitting it later
# (e.g. cheap DB work vs. expensive LLM work) is a worker-config change, not a code one.
TEMPORAL_TASK_QUEUE: str = os.getenv("TEMPORAL_TASK_QUEUE", "agentshield")

# --- Concurrency -------------------------------------------------------------
# How many agents one batch crash-tests at the same time. Under Temporal this is passed
# into RunGroupWorkflow and enforced there, so it bounds the children of ONE batch — it is
# no longer a process-wide cap. WORK_CONCURRENCY below is what bounds total load globally.
AGENT_CONCURRENCY: int = int(os.getenv("AGENT_CONCURRENCY", "3"))
# Ceiling on in-flight scenario/judge/fix work, applied by the Temporal worker as
# max_concurrent_activities. Every workflow on that worker draws from this one pool, so it
# is a genuinely global limit — stronger than the per-process asyncio semaphore it
# replaced. Without it, N concurrent agent tests would each open their own window onto the
# LLM and the agents under test.
#
# Set to 3 x AGENT_CONCURRENCY, so that even at maximum fan-out every agent gets 3
# concurrent scenarios rather than a token 1 or 2. Measured against the sample agents:
# median response time is flat from 1 to 9 concurrent requests (~0.9s) and only the tail
# starts moving at 12 (max 3.8s). Since the fair share means ONE agent never sees more
# than WORK_CONCURRENCY requests at a time, 9 keeps per-agent load inside that flat zone
# — which matters because load-induced timeouts would show up as the agent's failures,
# not ours. Lower it for a slower or more fragile agent under test.
WORK_CONCURRENCY: int = int(os.getenv("WORK_CONCURRENCY", "9"))

# How many workflow *decisions* a worker may process at once. This does NOT limit how many
# agent tests run — a workflow waiting on an activity holds no slot at all. Keep it
# generous: setting it low throttles the scheduling of work without capping the work
# itself, which starves concurrency instead of bounding it. WORK_CONCURRENCY is the knob
# that bounds actual load.
WORKFLOW_TASK_CONCURRENCY: int = int(os.getenv("WORKFLOW_TASK_CONCURRENCY", "100"))

# Rough $ per 1K tokens for the agent-under-test, used to estimate run cost.
# Default is a blended gpt-4o-mini rate; override per model via .env.
PRICE_PER_1K_TOKENS: float = float(os.getenv("PRICE_PER_1K_TOKENS", "0.0004"))

# --- Voice (Phase 2B: AI Caller) ---------------------------------------------
# STT/TTS models app.core.voice_caller uses to bridge a text scenario turn onto a real
# voice-contract endpoint. Same OpenAI vendor as every other model call in this app —
# no new provider. Defaults match backend/sample_voice_bot's own defaults.
VOICE_STT_MODEL: str = os.getenv("VOICE_STT_MODEL", "whisper-1")
VOICE_TTS_MODEL: str = os.getenv("VOICE_TTS_MODEL", "tts-1")
VOICE_TTS_VOICE: str = os.getenv("VOICE_TTS_VOICE", "alloy")

# --- Voice (Phase 3A: Twilio, one call per turn) ------------------------------
# Twilio account credentials + the number AgentShield places outbound test calls
# from. PUBLIC_BASE_URL is AgentShield's OWN publicly reachable address (e.g. an
# ngrok URL in dev) — Twilio calls back to it (webhook + Media Stream), the reverse
# direction from every other transport, which always calls OUT to the agent.
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "")

# --- Agent name mapping (backend/mapping.yaml) -------------------------------
# Maps the internal agent identifier used in the repo (Domain.key, or the module name
# for the standalone sample_rag_bot) to the agent's actual / display name.
MAPPING_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mapping.yaml")


def _load_agent_names() -> dict[str, str]:
    """Read mapping.yaml once at import. A missing/invalid file is non-fatal."""
    try:
        with open(MAPPING_PATH) as f:
            data = yaml.safe_load(f) or {}
        agents = data.get("agents") or {}
        return {str(k): str(v) for k, v in agents.items()}
    except Exception as e:
        print(f"[config] could not read {MAPPING_PATH} ({e}); using internal names as-is")
        return {}


AGENT_NAMES: dict[str, str] = _load_agent_names()


def agent_display_name(key: str) -> str:
    """The agent's mapped name, falling back to the internal identifier if unmapped."""
    return AGENT_NAMES.get(key, key)


# --- Customer inventory (backend/inventory.yaml) -----------------------------
# Which agents are onboarded for each customer. Agent entries are the same internal
# identifiers used as mapping.yaml keys, so display names come from AGENT_NAMES.
INVENTORY_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inventory.yaml")


def _load_inventory() -> list[dict]:
    """Read inventory.yaml once at import. A missing/invalid file is non-fatal."""
    try:
        with open(INVENTORY_PATH) as f:
            data = yaml.safe_load(f) or {}
        customers = data.get("customers") or []
        return [
            {"name": str(c.get("name", "")), "agents": [str(a) for a in (c.get("agents") or [])]}
            for c in customers
        ]
    except Exception as e:
        print(f"[config] could not read {INVENTORY_PATH} ({e}); inventory is empty")
        return []


INVENTORY: list[dict] = _load_inventory()


# Where the sample agents from backend/sample_agents/ listen, keyed by the same internal
# identifier used in mapping.yaml / inventory.yaml. Used to seed their `agents` rows.
SAMPLE_AGENT_URLS: dict[str, str] = {
    "banking_bot": os.getenv("BANKING_BOT_URL", "http://localhost:8003/chat"),
    "hr_bot": os.getenv("HR_BOT_URL", "http://localhost:8004/chat"),
    "insurance_bot": os.getenv("INSURANCE_BOT_URL", "http://localhost:8005/chat"),
    "airline_bot": os.getenv("AIRLINE_BOT_URL", "http://localhost:8006/chat"),
    "telecom_bot": os.getenv("TELECOM_BOT_URL", "http://localhost:8007/chat"),
    "sample_rag_bot": SAMPLE_RAG_BOT_URL,
}
