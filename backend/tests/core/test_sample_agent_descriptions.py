"""Sample agent descriptions live in backend/mapping.yaml (`descriptions:`), not in code.
Every sample agent the repo seeds must have one: it is what scenario generation, the Judge
and the fixer know about the agent when no docs are uploaded."""
from app.config import AGENT_DESCRIPTIONS, INVENTORY, agent_description
from app.core.adapter import SAMPLE_RAG_BOT_KEY


def test_every_seeded_sample_agent_has_a_description():
    seeded = {SAMPLE_RAG_BOT_KEY} | {key for customer in INVENTORY for key in customer["agents"]}
    missing = sorted(seeded - set(AGENT_DESCRIPTIONS))
    assert not missing, f"add these to mapping.yaml's descriptions: {missing}"
    assert all(AGENT_DESCRIPTIONS[k].strip() for k in seeded)


def test_descriptions_are_read_as_one_line():
    # Folded (>-) YAML blocks: wrapped lines are joined with single spaces.
    assert all("\n" not in d and "  " not in d for d in AGENT_DESCRIPTIONS.values())


def test_unmapped_agent_gets_a_placeholder():
    assert agent_description("no_such_bot") == "Sample agent 'no_such_bot'."
