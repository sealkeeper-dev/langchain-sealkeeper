"""SealGateInterceptor, a tool interceptor for langchain-mcp-adapters that
gates the tools of named MCP servers on each server's SEAL."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from langchain_core.messages import ToolMessage

from ._shared import blocked_text, default_gate


class SealGateInterceptor:
    """Checks a server's SEAL before each call to one of its tools.

    Pass it as MultiServerMCPClient(connections, tool_interceptors=[gate]),
    with the same connections. servers names the servers to gate. Each call
    to one of their tools asks the gate about that server's url, read from
    connections at the call, as a service. A pass calls the tool. A block
    answers an error ToolMessage with the code and the reason and never
    calls it, so the agent's run goes on. That ToolMessage has no name,
    and its tool_call_id is the one LangGraph's ToolNode passes the tool,
    empty for a call made outside one. A server not named passes untouched.
    A stdio server has no URL to check and is refused. The gate sees a
    tool's server only when the adapter loaded it with the server's name,
    as client.get_tools() does. allow_unsealed also passes a server
    SealKeeper does not know, one it is still checking and one with no SEAL
    for dormancy.
    """

    def __init__(
        self,
        connections: Mapping[str, Mapping[str, Any]],
        servers: Iterable[str],
        *,
        allow_unsealed: bool = False,
    ) -> None:
        if isinstance(servers, str):
            raise TypeError("servers must be a list of server names")
        names = tuple(servers)
        for name in names:
            connection = connections.get(name)
            if connection is None:
                raise ValueError(f"no connection is named {name!r}")
            if connection.get("transport") == "stdio":
                raise ValueError(f"{name!r} is a stdio server, with no URL to check")
        self._connections = connections
        self._servers = frozenset(names)
        self._gate = default_gate(allow_unsealed)

    async def __call__(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        if request.server_name not in self._servers:
            return await handler(request)
        connection = self._connections.get(request.server_name) or {}
        decision = await self._gate.acheck(connection.get("url"), "service")
        if decision.passed:
            return await handler(request)
        return ToolMessage(
            content=blocked_text(decision),
            tool_call_id=getattr(request.runtime, "tool_call_id", None) or "",
            status="error",
        )
