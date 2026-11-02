"""What the tool and both gates share. The User-Agent suffix, the gate both
gates decide with and the line a block answers with."""

from __future__ import annotations

from sealkeeper import AsyncSealKeeper, Block, Gate, SealKeeper

from ._version import __version__

USER_AGENT_SUFFIX = f"langchain-sealkeeper/{__version__}"


def default_gate(allow_unsealed: bool) -> Gate:
    """A Gate with its default rules and allow_unsealed, whose lookups name
    this package in their User-Agent."""
    return Gate(
        allow_unsealed=allow_unsealed,
        client=SealKeeper(user_agent_suffix=USER_AGENT_SUFFIX),
        async_client=AsyncSealKeeper(user_agent_suffix=USER_AGENT_SUFFIX),
    )


def blocked_text(block: Block) -> str:
    """The error text of a blocked call, its code and its reason."""
    return f"SealKeeper blocked this call: {block.code}. {block.reason}"
