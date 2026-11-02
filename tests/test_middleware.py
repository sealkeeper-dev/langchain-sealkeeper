"""SealGateMiddleware under create_agent, with a fake tool-calling model
and a recorded SealKeeper, invoked and awaited."""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from harness import (
    MCP_URL,
    RecordedSealKeeper,
    calls_then_done,
)
from langchain_sealkeeper import SealGateMiddleware, SealTarget, __version__

pytestmark = pytest.mark.anyio
MODES = ["sync", "async"]


class Setup:
    """An agent with two tools that record their calls. ask_remote is pinned
    to target, search is not."""

    def __init__(
        self,
        target: SealTarget,
        sealkeeper: RecordedSealKeeper,
        allow_unsealed: bool = False,
    ) -> None:
        self.called: list[str] = []
        self.sealkeeper = sealkeeper
        self.middleware = SealGateMiddleware(
            {"ask_remote": target}, allow_unsealed=allow_unsealed
        )

        @tool
        def ask_remote(text: str) -> str:
            """Sends text to the remote agent."""
            self.called.append("ask_remote")
            return f"remote says {text}"

        @tool
        def search(text: str) -> str:
            """Searches for text."""
            self.called.append("search")
            return f"found {text}"

        self.tools = [ask_remote, search]

    async def run(self, mode: str, name: str) -> list[BaseMessage]:
        """One run whose model calls name once, then answers done."""
        agent = create_agent(
            calls_then_done(name, {"text": "hi"}),
            tools=self.tools,
            middleware=[self.middleware],
        )
        state = {"messages": [HumanMessage("go")]}
        if mode == "sync":
            return agent.invoke(state)["messages"]
        return (await agent.ainvoke(state))["messages"]


def tool_message(messages: list[BaseMessage]) -> ToolMessage:
    [found] = [m for m in messages if isinstance(m, ToolMessage)]
    return found


@pytest.mark.parametrize("mode", MODES)
async def test_a_pinned_tool_whose_target_passes_is_called(mode, sealkeeper):
    setup = Setup(SealTarget("agent", "alice/helper"), sealkeeper)
    messages = await setup.run(mode, "ask_remote")
    found = tool_message(messages)
    assert (found.status, found.content) == ("success", "remote says hi")
    assert setup.called == ["ask_remote"]
    assert setup.sealkeeper.targets() == ["alice/helper"]
    assert messages[-1].content == "done"


@pytest.mark.parametrize("mode", MODES)
async def test_a_block_answers_an_error_tool_message_and_the_run_goes_on(
    mode, sealkeeper
):
    setup = Setup(SealTarget("agent", "alice/broken"), sealkeeper)
    messages = await setup.run(mode, "ask_remote")
    found = tool_message(messages)
    assert (found.status, found.tool_call_id, found.name) == (
        "error",
        "call-1",
        "ask_remote",
    )
    assert found.content == (
        "SealKeeper blocked this call: bad_signature. broken SEAL: bad_signature"
    )
    assert setup.called == []
    assert messages[-1].content == "done"


@pytest.mark.parametrize("mode", MODES)
async def test_a_target_of_the_other_kind_blocks(mode, sealkeeper):
    setup = Setup(SealTarget("service", "alice/helper"), sealkeeper)
    found = tool_message(await setup.run(mode, "ask_remote"))
    assert found.status == "error"
    assert found.content.startswith("SealKeeper blocked this call: wrong_kind. ")
    assert setup.called == []


@pytest.mark.parametrize("mode", MODES)
async def test_sealkeeper_unavailable_blocks_and_the_run_goes_on(mode, sealkeeper):
    sealkeeper.mcp_status = 503
    setup = Setup(SealTarget("agent", "alice/helper"), sealkeeper)
    messages = await setup.run(mode, "ask_remote")
    found = tool_message(messages)
    assert found.status == "error"
    assert found.content.startswith("SealKeeper blocked this call: unavailable. ")
    assert setup.called == []
    assert messages[-1].content == "done"


@pytest.mark.parametrize("mode", MODES)
async def test_allow_unsealed_calls_a_target_sealkeeper_does_not_know(mode, sealkeeper):
    setup = Setup(SealTarget("agent", "alice/unknown"), sealkeeper, True)
    found = tool_message(await setup.run(mode, "ask_remote"))
    assert (found.status, found.content) == ("success", "remote says hi")
    assert setup.called == ["ask_remote"]


@pytest.mark.parametrize("mode", MODES)
async def test_allow_unsealed_still_blocks_a_broken_seal(mode, sealkeeper):
    setup = Setup(SealTarget("agent", "alice/broken"), sealkeeper, True)
    found = tool_message(await setup.run(mode, "ask_remote"))
    assert found.content.startswith("SealKeeper blocked this call: bad_signature. ")
    assert setup.called == []


@pytest.mark.parametrize("mode", MODES)
async def test_a_tool_not_pinned_passes_untouched(mode, sealkeeper):
    setup = Setup(SealTarget("agent", "alice/broken"), sealkeeper)
    found = tool_message(await setup.run(mode, "search"))
    assert (found.status, found.content) == ("success", "found hi")
    assert setup.called == ["search"]
    assert setup.sealkeeper.requests == []


def test_a_target_is_an_agent_or_a_service_with_a_string_ref():
    with pytest.raises(ValueError, match='kind must be "agent" or "service"'):
        SealTarget("robot", "alice/helper")
    with pytest.raises(TypeError):
        SealTarget("agent", None)
    with pytest.raises(TypeError):
        SealGateMiddleware({"ask_remote": ("agent", "alice/helper")})


@pytest.mark.parametrize("mode", MODES)
async def test_the_gate_names_this_package_in_its_user_agent(mode, sealkeeper):
    setup = Setup(SealTarget("agent", "alice/helper"), sealkeeper)
    assert tool_message(await setup.run(mode, "ask_remote")).status == "success"
    [request] = [r for r in sealkeeper.requests if str(r.url) == MCP_URL]
    assert request.headers["user-agent"].endswith(
        f" langchain-sealkeeper/{__version__}"
    )
