"""SealGateMiddleware, agent middleware that gates the tools which call a
remote agent or service on that target's SEAL."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from sealkeeper import Block

from ._shared import blocked_text, default_gate

if TYPE_CHECKING:
    # The type langchain's own hooks answer with.
    from langgraph.types import Command


@dataclass(frozen=True)
class SealTarget:
    """What a gated tool calls. kind is agent or service, ref the target to
    look up, such as a handle, a card URL or an MCP server URL."""

    kind: Literal["agent", "service"]
    ref: str

    def __post_init__(self) -> None:
        if self.kind not in ("agent", "service"):
            raise ValueError('kind must be "agent" or "service"')
        if not isinstance(self.ref, str):
            raise TypeError("ref must be a string")


class SealGateMiddleware(AgentMiddleware):
    """Checks a target's SEAL before each call to a tool pinned to it.

    targets maps a tool name to the SealTarget it calls. A pass calls the
    tool. A block answers an error ToolMessage with the code and the reason
    and never calls it, so the agent's run goes on. A tool not pinned passes
    untouched. allow_unsealed also passes a target SealKeeper does not
    know, one it is still checking and one with no SEAL for dormancy.
    """

    def __init__(
        self, targets: Mapping[str, SealTarget], *, allow_unsealed: bool = False
    ) -> None:
        super().__init__()
        for target in targets.values():
            if not isinstance(target, SealTarget):
                raise TypeError("each target must be a SealTarget")
        self._targets = dict(targets)
        self._gate = default_gate(allow_unsealed)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        target = self._targets.get(request.tool_call["name"])
        if target is None:
            return handler(request)
        decision = self._gate.check(target.ref, target.kind)
        if decision.passed:
            return handler(request)
        return _blocked(request, decision)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        target = self._targets.get(request.tool_call["name"])
        if target is None:
            return await handler(request)
        decision = await self._gate.acheck(target.ref, target.kind)
        if decision.passed:
            return await handler(request)
        return _blocked(request, decision)


def _blocked(request: ToolCallRequest, block: Block) -> ToolMessage:
    return ToolMessage(
        content=blocked_text(block),
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
        status="error",
    )
