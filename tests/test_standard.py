"""LangChain's standard tool tests for SealKeeperLookupTool, over the
recorded SealKeeper of conftest.py."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_sealkeeper import SealKeeperLookupTool

# The standard async tests carry no mark of their own.
pytestmark = pytest.mark.anyio


class Recorded:
    @property
    def tool_constructor(self) -> type[SealKeeperLookupTool]:
        return SealKeeperLookupTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"target": "alice/helper"}


class TestLookupToolUnit(Recorded, ToolsUnitTests):
    pass


class TestLookupToolIntegration(Recorded, ToolsIntegrationTests):
    pass
