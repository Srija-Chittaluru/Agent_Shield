"""Unit tests for app.core.activities.prepare_scenarios' wiring of topic extraction and
`existing` passthrough into app.core.scenarios.generate_scenarios — both mocked, so this
verifies the plumbing only, not real LLM behavior.
"""
from app.core import activities


async def test_prepare_scenarios_extracts_topics_and_forwards_existing(monkeypatch):
    captured = {}

    async def fake_extract_topics(knowledge):
        captured["knowledge_for_topics"] = knowledge
        return ["patient verification", "packet redelivery"]

    async def fake_generate_scenarios(description, selected_types=None, guidance="",
                                       knowledge="", topics=None, existing=None):
        captured["topics"] = topics
        captured["existing"] = existing
        return [{"title": "x"}]

    monkeypatch.setattr(activities, "extract_topics", fake_extract_topics)
    monkeypatch.setattr(activities, "generate_scenarios", fake_generate_scenarios)

    agent = {"id": 1, "knowledge": "Maya's docs", "description": ""}
    existing = [{"title": "Old one"}]
    result = await activities.prepare_scenarios(agent, ["support"], "", "", existing)

    assert captured["knowledge_for_topics"] == "Maya's docs"
    assert captured["topics"] == ["patient verification", "packet redelivery"]
    assert captured["existing"] == existing
    assert result == [{"title": "x"}]


async def test_prepare_scenarios_existing_defaults_to_none(monkeypatch):
    captured = {}

    async def fake_extract_topics(knowledge):
        return []

    async def fake_generate_scenarios(description, selected_types=None, guidance="",
                                       knowledge="", topics=None, existing=None):
        captured["existing"] = existing
        return []

    monkeypatch.setattr(activities, "extract_topics", fake_extract_topics)
    monkeypatch.setattr(activities, "generate_scenarios", fake_generate_scenarios)

    await activities.prepare_scenarios({"id": 1, "knowledge": "docs", "description": ""}, [])
    assert captured["existing"] is None
