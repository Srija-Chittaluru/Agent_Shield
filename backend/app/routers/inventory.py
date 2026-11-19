"""/inventory API — the customer -> agents combinations, and their stored test cases.

Sample combinations come from backend/inventory.yaml (seeded into PostgreSQL at startup);
agents connected through "Connect Your AI Agent" are appended straight from PostgreSQL,
which is the source of truth for them. Each agent carries its `customer_agent_id` — the id
everything downstream keys off, because the same agent onboarded for two customers is two
separate testing contexts.
"""
from fastapi import APIRouter, HTTPException

from app.config import INVENTORY, agent_display_name
from app.core.mock_agent import MOCK_KIND
from app.db import get_customer_agent, get_test_cases, list_customer_agents

router = APIRouter(prefix="/inventory", tags=["inventory"])


@router.get("")
def get_inventory() -> dict:
    """Customers with their onboarded agents (internal key, display name, and DB ids)."""
    # (customer name, agent name) -> the customer_agents row.
    combos = {(c["customer_name"], c["agent_name"]): c for c in list_customer_agents()}
    customers = []
    seeded: set[tuple[str, str]] = set()
    for c in INVENTORY:
        agents = []
        for key in c["agents"]:
            name = agent_display_name(key)
            combo = combos.get((c["name"], name))
            seeded.add((c["name"], name))
            agents.append({
                "key": key,
                "name": name,
                "customer_agent_id": combo["id"] if combo else None,
                "agent_id": combo["agent_id"] if combo else None,
                # Unregistered inventory.yaml entries (combo is None) have no agents row
                # to read modality from yet — default to "chat", same as the DB column.
                "modality": combo["agent_modality"] if combo else "chat",
            })
        customers.append({"name": c["name"], "agents": agents})

    # Agents connected through "Connect Your AI Agent" exist only in PostgreSQL — they have
    # no inventory.yaml entry — so append every combination the seeded loop didn't emit.
    # Their display name is the agents.name column, so mapping.yaml needs no entry either.
    dynamic: dict[str, list[dict]] = {}
    for (customer_name, agent_name), combo in combos.items():
        if (customer_name, agent_name) in seeded:
            continue
        dynamic.setdefault(customer_name, []).append({
            "key": f"agent-{combo['agent_id']}",
            "name": agent_name,
            "customer_agent_id": combo["id"],
            "agent_id": combo["agent_id"],
            "modality": combo["agent_modality"],
            # Dry-run agents are stored like any connected agent; this lets the UI label them.
            "simulated": combo["agent_kind"] == MOCK_KIND,
        })
    # dict order follows list_customer_agents()' ORDER BY, i.e. the order they were created.
    customers.extend({"name": name, "agents": agents} for name, agents in dynamic.items())
    return {"customers": customers}


@router.get("/{customer_agent_id}/test-cases")
def get_stored_test_cases(customer_agent_id: int) -> dict:
    """The test cases saved for ONE customer-agent combination (empty list if none yet).

    Feeds the same Review Test Cases UI that newly generated cases go through.
    """
    import json

    combo = get_customer_agent(customer_agent_id)
    if combo is None:
        raise HTTPException(status_code=404, detail=f"customer-agent {customer_agent_id} not found")

    cases = []
    for row in get_test_cases(customer_agent_id):
        try:
            seed_turns = json.loads(row["seed_turns_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            seed_turns = []
        cases.append({
            "title": row["title"],
            "user_goal": row["user_goal"],
            "test_type": row["test_type"],
            "assigned_fault": row["assigned_fault"],
            "expected_behavior": row["expected_behavior"],
            "seed_turns": seed_turns,
            "source": row["source"] or "ai",
        })

    return {
        "customer_agent_id": combo["id"],
        "customer_name": combo["customer_name"],
        "agent_id": combo["agent_id"],
        "agent_name": combo["agent_name"],
        "scenarios": cases,
    }
