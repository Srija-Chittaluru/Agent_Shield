#!/usr/bin/env bash
# AgentShield — start everything with one command.
#   ./run_all.sh
# Starts: backend (8100), 6 sample agents (8002-8007), and the frontend (3100).
# Ctrl-C stops them all.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BE="$ROOT/backend"
FE="$ROOT/frontend"
PY="$BE/.venv/bin/uvicorn"
export PATH="/opt/homebrew/bin:$PATH"   # ensure node/npm are found (Homebrew)

pids=()
cleanup() {
  echo ""
  echo "Stopping AgentShield…"
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null; done
  # free the ports in case anything lingers
  for p in 8100 8002 8003 8004 8005 8006 8007 3100 7233 8233; do lsof -ti:$p | xargs kill 2>/dev/null; done
  exit 0
}
trap cleanup INT TERM

if [ ! -x "$PY" ]; then
  echo "ERROR: backend venv not found at $PY"
  echo "Run:  cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [ ! -f "$BE/.env" ]; then
  echo "WARNING: $BE/.env missing — copy .env.example to .env and paste your OPENAI_API_KEY."
fi

start_agent() { # name module port
  echo "  → $1 on :$3"
  ( cd "$BE" && "$PY" "$2" --port "$3" --log-level warning ) &
  pids+=($!)
}

echo "Starting AgentShield services…"
start_agent "Backend (AgentShield API)"      "app.main:app"                 8100
start_agent "Store Support agent"            "sample_rag_bot.main:app"      8002
start_agent "NorthBank (banking) agent"      "sample_agents.banking:app"    8003
start_agent "PeopleDesk (HR) agent"          "sample_agents.hr:app"         8004
start_agent "SafeGuard (insurance) agent"    "sample_agents.insurance:app"  8005
start_agent "SkyRoute (airline) agent"       "sample_agents.airline:app"    8006
start_agent "ConnectWave (telecom) agent"    "sample_agents.telecom:app"    8007

# --- Temporal (optional) -----------------------------------------------------
# Durable workflow execution. Nothing in the app depends on it yet, so a machine without
# the CLI simply skips this and everything else behaves exactly as before.
# The dev server keeps its state in memory: stop it and history is gone. That is fine for
# development; add --db-filename to persist across restarts.
if command -v temporal >/dev/null 2>&1; then
  echo "  → Temporal dev server on :7233 (UI on :8233)"
  ( temporal server start-dev --ip 127.0.0.1 --log-level error ) &
  pids+=($!)
  sleep 3   # the worker needs the server accepting connections before it polls
  echo "  → Temporal worker"
  ( cd "$BE" && "$BE/.venv/bin/python" -m app.temporal.worker ) &
  pids+=($!)
else
  echo "  → Temporal: 'temporal' CLI not found — skipping (brew install temporal)"
fi

echo "  → Frontend on :3100"
( cd "$FE" && npm run dev ) &
pids+=($!)

sleep 4
cat <<EOF

────────────────────────────────────────────────────────────
AgentShield is up:
  • App (open this):     http://localhost:3100
  • Backend API:         http://localhost:8100/health
  Agents under test (each has its own chat UI at its root):
  • Store Support:       http://localhost:8002/
  • NorthBank (banking): http://localhost:8003/
  • PeopleDesk (HR):     http://localhost:8004/
  • SafeGuard (insurance): http://localhost:8005/
  • SkyRoute (airline):  http://localhost:8006/
  • ConnectWave (telecom): http://localhost:8007/

Temporal (workflow engine, not yet on the request path):
  • Web UI:              http://localhost:8233

Knowledge docs to upload live in:  docs/agent_kbs/
Press Ctrl-C to stop everything.
────────────────────────────────────────────────────────────
EOF

wait
