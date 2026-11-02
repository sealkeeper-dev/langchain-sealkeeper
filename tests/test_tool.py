"""SealKeeperLookupTool, its line, its artifact and its errors, invoked and
awaited, alone and under create_agent with a checkpointer."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from harness import (
    HELPER_ID,
    MCP_URL,
    NOW,
    calls_then_done,
)
from langchain_sealkeeper import SealKeeperLookupTool, __version__

pytestmark = pytest.mark.anyio
MODES = ["sync", "async"]


async def call(mode: str, target: str):
    """The ToolMessage of one call of the tool with a tool call id."""
    lookup = SealKeeperLookupTool()
    tool_call = {
        "name": "sealkeeper_lookup",
        "args": {"target": target},
        "id": "call-1",
        "type": "tool_call",
    }
    if mode == "sync":
        return lookup.invoke(tool_call)
    return await lookup.ainvoke(tool_call)


@pytest.mark.parametrize("mode", MODES)
async def test_a_verified_seal_reads_its_kind_level_and_trust(mode):
    message = await call(mode, "alice/helper")
    assert isinstance(message, ToolMessage)
    assert (message.status, message.content) == (
        "success",
        "alice/helper: verified agent SEAL, level bronze, Trust Score 412.",
    )
    artifact = message.artifact
    assert (artifact["verified"], artifact["status"], artifact["code"]) == (
        True,
        "shown",
        None,
    )
    seal = artifact["seal"]
    assert (seal["agent_id"], seal["kind"], seal["level"], seal["trust"]) == (
        HELPER_ID,
        "agent",
        "bronze",
        412,
    )
    assert seal["expires_at"] == NOW - 3600 + 86400
    assert seal["payload"]["scores"] == {"reliability": 0.9}


@pytest.mark.parametrize("mode", MODES)
async def test_a_broken_seal_reads_verified_false_and_never_its_level(mode):
    message = await call(mode, "alice/broken")
    assert (message.status, message.content) == (
        "success",
        "alice/broken: no verified SEAL (broken, bad_signature).",
    )
    artifact = message.artifact
    assert (artifact["verified"], artifact["seal"], artifact["code"]) == (
        False,
        None,
        "broken",
    )
    assert artifact["broken"] == "bad_signature"
    # The answer's unsigned fields claim gold and 999, and its SEAL's
    # changed payload claims gold. None of it reaches the model or the
    # artifact.
    for text in (message.content, json.dumps(artifact)):
        assert "gold" not in text
        assert "999" not in text
        assert "bronze" not in text


@pytest.mark.parametrize("mode", MODES)
async def test_a_target_sealkeeper_does_not_know_reads_not_found(mode):
    message = await call(mode, "alice/nobody")
    assert message.content == "alice/nobody: no verified SEAL (not_found)."
    assert (message.artifact["verified"], message.artifact["code"]) == (
        False,
        "not_found",
    )


@pytest.mark.parametrize("mode", MODES)
async def test_a_lookup_that_fails_answers_an_error_tool_message(mode, sealkeeper):
    sealkeeper.mcp_status = 503
    message = await call(mode, "alice/helper")
    assert message.status == "error"
    assert message.tool_call_id == "call-1"
    assert "SealKeeper" in message.content
    assert message.artifact is None


@pytest.mark.parametrize("mode", MODES)
async def test_the_artifact_holds_only_json_values(mode):
    for target in ("alice/helper", "alice/broken", "alice/nobody"):
        artifact = (await call(mode, target)).artifact
        assert json.loads(json.dumps(artifact)) == artifact


@pytest.mark.parametrize("mode", MODES)
async def test_the_artifact_survives_a_checkpointer_that_allows_only_safe_types(
    mode,
):
    expected = (await call(mode, "alice/helper")).artifact
    # allowed_msgpack_modules None revives only LangGraph's safe types.
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=None))
    agent = create_agent(
        calls_then_done("sealkeeper_lookup", {"target": "alice/helper"}),
        tools=[SealKeeperLookupTool()],
        checkpointer=saver,
    )
    config: Any = {"configurable": {"thread_id": "one"}}
    state = {"messages": [HumanMessage("go")]}
    if mode == "sync":
        agent.invoke(state, config)
        stored = agent.get_state(config)
    else:
        await agent.ainvoke(state, config)
        stored = await agent.aget_state(config)
    messages = stored.values["messages"]
    [message] = [m for m in messages if isinstance(m, ToolMessage)]
    assert message.artifact == expected
    assert messages[-1].content == "done"


@pytest.mark.parametrize("mode", MODES)
async def test_a_failed_lookup_leaves_the_agent_run_going(mode, sealkeeper):
    sealkeeper.mcp_status = 503
    agent = create_agent(
        calls_then_done("sealkeeper_lookup", {"target": "alice/helper"}),
        tools=[SealKeeperLookupTool()],
    )
    state = {"messages": [HumanMessage("go")]}
    if mode == "sync":
        messages = agent.invoke(state)["messages"]
    else:
        messages = (await agent.ainvoke(state))["messages"]
    [message] = [m for m in messages if isinstance(m, ToolMessage)]
    assert message.status == "error"
    assert messages[-1].content == "done"


@pytest.mark.parametrize("mode", MODES)
async def test_a_client_made_names_this_package_in_its_user_agent(mode, sealkeeper):
    lookup = SealKeeperLookupTool()
    if mode == "sync":
        lookup.invoke({"target": "alice/helper"})
    else:
        await lookup.ainvoke({"target": "alice/helper"})
    [request] = [r for r in sealkeeper.requests if str(r.url) == MCP_URL]
    assert request.headers["user-agent"].endswith(
        f" langchain-sealkeeper/{__version__}"
    )


def test_the_tool_is_sealkeeper_lookup_with_one_string_argument():
    lookup = SealKeeperLookupTool()
    assert lookup.name == "sealkeeper_lookup"
    assert lookup.response_format == "content_and_artifact"
    schema = lookup.get_input_schema().model_json_schema()
    assert schema["required"] == ["target"]
    assert schema["properties"]["target"]["type"] == "string"
