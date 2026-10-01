"""Test doubles for app.core.runner._reactive_scripted_line's chat() call — no real
LLM, no real network.

Shared by every scripted-flow runner test (test_run_scenario_flow_node.py,
test_run_scenario_flow_path.py, test_run_scenario_interrupt.py,
test_run_scenario_session_end.py) that needs `_run_scripted` to keep sending its
seed_turns lines verbatim: `passthrough_chat` extracts "YOUR PLANNED NEXT LINE: ..."
from the prompt _reactive_scripted_line builds and echoes it straight back, so the
reactive rewrite becomes a deterministic no-op instead of a real (and non-deterministic)
model call.
"""
import re

_PLANNED_LINE = re.compile(r"YOUR PLANNED NEXT LINE: (.*)\n\nWrite", re.S)


async def passthrough_chat(system, messages, json_mode=True):
    content = messages[0]["content"]
    m = _PLANNED_LINE.search(content)
    return {"line": m.group(1).strip() if m else ""}
