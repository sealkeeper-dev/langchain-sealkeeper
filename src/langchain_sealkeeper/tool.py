"""SealKeeperLookupTool, a LangChain tool that looks a target up on
SealKeeper and verifies its SEAL."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, PrivateAttr
from sealkeeper import AsyncSealKeeper, Lookup, SealKeeper, SealKeeperError

from ._shared import USER_AGENT_SUFFIX


class LookupInput(BaseModel):
    target: str = Field(
        description=(
            "The agent or service to look up. A handle such as alice/helper, "
            "a SealKeeper page URL, an agent id, a did:key, an A2A card URL "
            "or an MCP server URL."
        )
    )


class SealKeeperLookupTool(BaseTool):
    """Looks a target up on SealKeeper and verifies its SEAL.

    The model reads one short line. The artifact is the lookup as a dict
    that holds only JSON values, so a checkpointer stores it as it is.
    level, kind and trust come only from a SEAL that verified, so a broken
    SEAL reads verified false and none of them. A lookup that fails answers
    an error ToolMessage with SealKeeper's line.
    """

    name: str = "sealkeeper_lookup"
    description: str = (
        "Look up an AI agent or service on SealKeeper and verify its SEAL, "
        "SealKeeper's signed credential. Answers whether a SEAL verified, "
        "and its kind, level and Trust Score when one did."
    )
    args_schema: type[BaseModel] = LookupInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    handle_tool_error: bool = True
    _client: SealKeeper | None = PrivateAttr(default=None)
    _async_client: AsyncSealKeeper | None = PrivateAttr(default=None)

    def _run(self, target: str) -> tuple[str, dict[str, Any]]:
        if self._client is None:
            self._client = SealKeeper(user_agent_suffix=USER_AGENT_SUFFIX)
        try:
            found = self._client.lookup(target)
        except SealKeeperError as error:
            raise ToolException(str(error)) from None
        return line_of(target, found), artifact_of(found)

    async def _arun(self, target: str) -> tuple[str, dict[str, Any]]:
        if self._async_client is None:
            self._async_client = AsyncSealKeeper(user_agent_suffix=USER_AGENT_SUFFIX)
        try:
            found = await self._async_client.lookup(target)
        except SealKeeperError as error:
            raise ToolException(str(error)) from None
        return line_of(target, found), artifact_of(found)


def line_of(target: str, found: Lookup) -> str:
    """The one line the model reads."""
    if found.verified:
        trust = "" if found.trust is None else f", Trust Score {found.trust}"
        return f"{target}: verified {found.kind} SEAL, level {found.level}{trust}."
    why = found.code if found.broken is None else f"{found.code}, {found.broken}"
    return f"{target}: no verified SEAL ({why})."


def artifact_of(found: Lookup) -> dict[str, Any]:
    """The lookup as a dict of JSON values, with verified beside its fields."""
    return {"verified": found.verified, **dataclasses.asdict(found)}
