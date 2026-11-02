"""SealGateInterceptor through a real MultiServerMCPClient, against
recorded MCP servers and a recorded SealKeeper."""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from harness import (
    MCP_URL,
    OTHER_URL,
    SERVER_URL,
    RecordedMcpServer,
    RecordedSealKeeper,
    calls_then_done,
    http_connection,
)
from langchain_sealkeeper import SealGateInterceptor, __version__

pytestmark = pytest.mark.anyio
NOT_FOUND_TEXT = (
    "SealKeeper blocked this call: not_found. "
    "SealKeeper knows no agent or service by that target"
)


class Setup:
    """Two recorded MCP servers. remote, at SERVER_URL, holds a service SEAL
    for its URL. other, at OTHER_URL, is unknown to SealKeeper."""

    def __init__(
        self,
        servers: Iterable[str],
        sealkeeper: RecordedSealKeeper,
        allow_unsealed: bool = False,
    ) -> None:
        self.remote = RecordedMcpServer("forecast")
        self.other = RecordedMcpServer("notes")
        self.connections = {
            "remote": http_connection(SERVER_URL, self.remote),
            "other": http_connection(OTHER_URL, self.other),
        }
        self.sealkeeper = sealkeeper
        self.interceptor = SealGateInterceptor(
            self.connections, servers, allow_unsealed=allow_unsealed
        )
        self.client = MultiServerMCPClient(
            self.connections, tool_interceptors=[self.interceptor]
        )

    async def run(self, tool: str) -> list[BaseMessage]:
        """One agent run whose model calls tool once, then answers done."""
        agent = create_agent(
            calls_then_done(tool, {"text": "hi"}), tools=await self.client.get_tools()
        )
        result = await agent.ainvoke({"messages": [HumanMessage("go")]})
        return result["messages"]


def tool_message(messages: list[BaseMessage]) -> ToolMessage:
    [found] = [m for m in messages if isinstance(m, ToolMessage)]
    return found


def texts(message: ToolMessage) -> list[str]:
    return [block["text"] for block in message.content]


async def test_a_gated_server_with_a_seal_for_its_url_is_called(sealkeeper):
    setup = Setup(["remote"], sealkeeper)
    messages = await setup.run("forecast")
    found = tool_message(messages)
    assert (found.status, texts(found)) == ("success", ["echo hi"])
    assert setup.remote.methods.count("tools/call") == 1
    assert setup.sealkeeper.targets() == [SERVER_URL]
    assert messages[-1].content == "done"


async def test_a_block_answers_an_error_tool_message_and_the_run_goes_on(sealkeeper):
    setup = Setup(["remote", "other"], sealkeeper)
    messages = await setup.run("notes")
    found = tool_message(messages)
    assert (found.status, found.tool_call_id, found.name, found.content) == (
        "error",
        "call-1",
        None,
        NOT_FOUND_TEXT,
    )
    assert "tools/call" not in setup.other.methods
    assert setup.sealkeeper.targets() == [OTHER_URL]
    assert messages[-1].content == "done"


async def test_sealkeeper_unavailable_blocks_and_the_run_goes_on(sealkeeper):
    sealkeeper.mcp_status = 503
    setup = Setup(["remote"], sealkeeper)
    messages = await setup.run("forecast")
    found = tool_message(messages)
    assert found.status == "error"
    assert found.content.startswith("SealKeeper blocked this call: unavailable. ")
    assert "tools/call" not in setup.remote.methods
    assert messages[-1].content == "done"


async def test_allow_unsealed_calls_a_server_sealkeeper_does_not_know(sealkeeper):
    setup = Setup(["other"], sealkeeper, allow_unsealed=True)
    found = tool_message(await setup.run("notes"))
    assert (found.status, texts(found)) == ("success", ["echo hi"])
    assert setup.sealkeeper.targets() == [OTHER_URL]


async def test_allow_unsealed_still_blocks_when_sealkeeper_is_unavailable(sealkeeper):
    sealkeeper.mcp_status = 503
    setup = Setup(["other"], sealkeeper, allow_unsealed=True)
    found = tool_message(await setup.run("notes"))
    assert found.content.startswith("SealKeeper blocked this call: unavailable. ")
    assert "tools/call" not in setup.other.methods


async def test_a_server_not_named_passes_untouched(sealkeeper):
    setup = Setup(["remote"], sealkeeper)
    messages = await setup.run("notes")
    found = tool_message(messages)
    assert (found.status, texts(found)) == ("success", ["echo hi"])
    assert setup.other.methods.count("tools/call") == 1
    assert setup.sealkeeper.requests == []


async def test_the_url_is_read_from_the_connections_the_client_uses(sealkeeper):
    setup = Setup(["remote"], sealkeeper)
    setup.connections["remote"]["url"] = OTHER_URL
    found = tool_message(await setup.run("forecast"))
    assert (found.status, found.content) == ("error", NOT_FOUND_TEXT)
    assert setup.sealkeeper.targets() == [OTHER_URL]


def test_a_stdio_server_named_is_refused():
    connections = {"local": {"transport": "stdio", "command": "notes", "args": []}}
    with pytest.raises(ValueError, match="'local' is a stdio server"):
        SealGateInterceptor(connections, ["local"])
    SealGateInterceptor(connections, [])


def test_a_name_with_no_connection_is_refused():
    with pytest.raises(ValueError, match="no connection is named 'remote'"):
        SealGateInterceptor({}, ["remote"])
    with pytest.raises(TypeError):
        SealGateInterceptor({"remote": {}}, "remote")


async def test_servers_given_as_a_generator_are_gated(sealkeeper):
    setup = Setup((name for name in ["other"]), sealkeeper)
    found = tool_message(await setup.run("notes"))
    assert (found.status, found.content) == ("error", NOT_FOUND_TEXT)
    assert "tools/call" not in setup.other.methods


async def test_the_gate_names_this_package_in_its_user_agent(sealkeeper):
    setup = Setup(["other"], sealkeeper)
    assert tool_message(await setup.run("notes")).status == "error"
    [request] = [r for r in sealkeeper.requests if str(r.url) == MCP_URL]
    assert request.headers["user-agent"].endswith(
        f" langchain-sealkeeper/{__version__}"
    )
