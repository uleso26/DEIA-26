# Imports.
from __future__ import annotations

import pytest

from agents.orchestrator import T2DOrchestrator, bootstrap_runtime
from data.canonical.resolver import get_resolver


# Bootstrap data.
@pytest.fixture(scope="session")
def bootstrap_data() -> None:
    bootstrap_runtime()


# Orchestrator.
@pytest.fixture(scope="session")
def orchestrator(bootstrap_data: None) -> T2DOrchestrator:
    runtime = T2DOrchestrator(bootstrap_if_needed=False)
    yield runtime
    runtime.close()


# MCP client.
@pytest.fixture(scope="session")
def mcp_client(orchestrator: T2DOrchestrator):
    return orchestrator.mcp_client


# Resolver.
@pytest.fixture(scope="session")
def resolver():
    return get_resolver()
