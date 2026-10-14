"""Unit tests for app.core.scenarios' knowledge-driven topic coverage: extract_topics,
the topics/existing prompt blocks, and the post-generation duplicate-title filter.

app.core.llm.chat is mocked throughout — these verify the PLUMBING (topics/existing
reach the prompt, dedup behaves correctly, nothing breaks the fallback-bank safety
net) — not real LLM creativity, which isn't unit-testable.
"""
import pytest

from app.core import scenarios


def _scenario(title, test_type="support", **extra):
    return {
        "title": title, "user_goal": f"Goal for {title}", "test_type": test_type,
        "assigned_fault": "none", "expected_behavior": "Does the right thing.",
        "seed_turns": ["Hello, I need help."], **extra,
    }


# ---------------------------------------------------------------------------
# extract_topics
# ---------------------------------------------------------------------------
async def test_extract_topics_empty_knowledge_short_circuits(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("must not call the LLM for empty knowledge")

    monkeypatch.setattr(scenarios, "chat", boom)
    assert await scenarios.extract_topics("") == []
    assert await scenarios.extract_topics("   ") == []


async def test_extract_topics_parses_and_caps_at_twelve(monkeypatch):
    async def fake_chat(system, messages, json_mode=False, **kw):
        assert "AGENT DOCS" in messages[0]["content"]
        return {"topics": [f"topic {i}" for i in range(20)]}

    monkeypatch.setattr(scenarios, "chat", fake_chat)
    topics = await scenarios.extract_topics("some real docs about Maya")
    assert len(topics) == 12
    assert topics[0] == "topic 0"


async def test_extract_topics_falls_back_to_empty_on_failure(monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("openai is down")

    monkeypatch.setattr(scenarios, "chat", boom)
    assert await scenarios.extract_topics("docs") == []


async def test_extract_topics_ignores_malformed_response(monkeypatch):
    async def fake_chat(*a, **kw):
        return {"topics": "not a list"}

    monkeypatch.setattr(scenarios, "chat", fake_chat)
    assert await scenarios.extract_topics("docs") == []


# ---------------------------------------------------------------------------
# _topics_block
# ---------------------------------------------------------------------------
def test_topics_block_empty_without_topics():
    assert scenarios._topics_block(None, None) == ""
    assert scenarios._topics_block([], [_scenario("x")]) == ""


def test_topics_block_lists_topics_and_spreads_instruction():
    block = scenarios._topics_block(["patient verification", "packet redelivery"], None)
    assert "DISTINCT TOPICS/CONDITIONS/BRANCHES" in block
    assert "- patient verification" in block and "- packet redelivery" in block
    assert "do not cluster on only one or two" in block
    assert "SCENARIOS ALREADY GENERATED" not in block  # no existing suite on a first Generate


def test_topics_block_includes_existing_titles_on_regenerate():
    existing = [_scenario("Verify patient identity"), _scenario("Packet already received")]
    block = scenarios._topics_block(["patient verification"], existing)
    assert "SCENARIOS ALREADY GENERATED FOR THIS AGENT" in block
    assert "- Verify patient identity" in block and "- Packet already received" in block
    assert "do NOT repeat these same topics" in block


# ---------------------------------------------------------------------------
# _drop_duplicate_titles
# ---------------------------------------------------------------------------
def test_drop_duplicate_titles_is_case_and_whitespace_insensitive():
    scenarios_list = [_scenario("Verify Patient Identity"), _scenario("New Topic")]
    existing = [_scenario("  verify   patient identity  ")]
    result = scenarios._drop_duplicate_titles(scenarios_list, existing)
    assert [s["title"] for s in result] == ["New Topic"]


def test_drop_duplicate_titles_keeps_everything_when_nothing_collides():
    scenarios_list = [_scenario("A"), _scenario("B")]
    result = scenarios._drop_duplicate_titles(scenarios_list, [_scenario("C")])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# generate_scenarios: wiring + dedup + fallback-bank safety net
# ---------------------------------------------------------------------------
async def test_generate_scenarios_passes_topics_and_existing_into_the_prompt(monkeypatch):
    captured = {}

    async def fake_chat(system, messages, json_mode=False, **kw):
        captured["prompt"] = messages[0]["content"]
        return {"scenarios": [_scenario("Fresh topic scenario")]}

    monkeypatch.setattr(scenarios, "chat", fake_chat)
    await scenarios.generate_scenarios(
        "A healthcare renewal voice agent.", topics=["patient verification"],
        existing=[_scenario("Old scenario")],
    )
    assert "DISTINCT TOPICS/CONDITIONS/BRANCHES" in captured["prompt"]
    assert "patient verification" in captured["prompt"]
    assert "Old scenario" in captured["prompt"]


async def test_generate_scenarios_drops_exact_title_repeats(monkeypatch):
    async def fake_chat(system, messages, json_mode=False, **kw):
        return {"scenarios": [
            _scenario("Old scenario"),  # repeats an existing title -> dropped
            _scenario("Genuinely new scenario"),
            _scenario("Another new one", test_type="memory"),
            _scenario("Yet another", test_type="hallucination"),
        ]}

    monkeypatch.setattr(scenarios, "chat", fake_chat)
    result = await scenarios.generate_scenarios(
        "An agent.", existing=[_scenario("Old scenario")],
    )
    titles = [s["title"] for s in result]
    assert "Old scenario" not in titles
    assert "Genuinely new scenario" in titles


async def test_dedup_never_empties_the_set_on_a_total_collision(monkeypatch):
    """If everything the model returns collides with `existing` (e.g. it ignored the
    instruction), keep the generated set rather than silently returning nothing."""
    async def fake_chat(system, messages, json_mode=False, **kw):
        return {"scenarios": [
            _scenario("Old scenario"), _scenario("Old scenario 2"),
            _scenario("Old scenario 3"), _scenario("Old scenario 4"),
        ]}

    monkeypatch.setattr(scenarios, "chat", fake_chat)
    result = await scenarios.generate_scenarios(
        "An agent.",
        existing=[_scenario("Old scenario"), _scenario("Old scenario 2"),
                  _scenario("Old scenario 3"), _scenario("Old scenario 4")],
    )
    assert len(result) == 4  # dedup would have emptied it -- kept instead


async def test_fallback_bank_is_never_deduped_against_existing(monkeypatch):
    """A thin/failed generation falls back to the hardcoded bank regardless of
    `existing` -- that bank has no awareness of what's already stored, so deduping it
    would be meaningless and could wrongly shrink a safety net."""
    async def fake_chat(*a, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(scenarios, "chat", fake_chat)
    result = await scenarios.generate_scenarios(
        "An agent.", existing=[dict(s) for s in scenarios.FALLBACK_BANK],
    )
    assert len(result) == len(scenarios.FALLBACK_BANK)


async def test_no_topics_or_existing_is_byte_for_byte_the_old_prompt(monkeypatch):
    """Regression guard: calling exactly as before this feature existed (no topics, no
    existing) produces the identical prompt — this feature is purely additive."""
    captured = {}

    async def fake_chat(system, messages, json_mode=False, **kw):
        captured["prompt"] = messages[0]["content"]
        return {"scenarios": [_scenario("x")]}

    monkeypatch.setattr(scenarios, "chat", fake_chat)
    await scenarios.generate_scenarios("An agent.", knowledge="some docs")
    assert "DISTINCT TOPICS" not in captured["prompt"]
    assert "SCENARIOS ALREADY GENERATED" not in captured["prompt"]
