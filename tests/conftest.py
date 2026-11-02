import pytest

from harness import RecordedSealKeeper, async_client, sync_client
from langchain_sealkeeper import _shared as shared
from langchain_sealkeeper import tool as tool_module


# Async tests run on asyncio, the loop httpx and LangChain use by default.
@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def sealkeeper(monkeypatch) -> RecordedSealKeeper:
    """The recorded SealKeeper behind every client the package makes, so no
    test reaches the network."""
    server = RecordedSealKeeper()
    for module in (shared, tool_module):
        monkeypatch.setattr(
            module, "SealKeeper", lambda **kw: sync_client(server, **kw)
        )
        monkeypatch.setattr(
            module, "AsyncSealKeeper", lambda **kw: async_client(server, **kw)
        )
    return server
