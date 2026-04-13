# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import pytest

from agents.orchestrator import T2DOrchestrator, bootstrap_runtime
from data.canonical.resolver import get_resolver


# Bootstrap the local test data once for the full test session
@pytest.fixture(scope="session")
def bootstrap_data() -> None:
    bootstrap_runtime()


# Provide a shared orchestrator fixture for the test suite
@pytest.fixture(scope="session")
def orchestrator(bootstrap_data: None) -> T2DOrchestrator:
    runtime = T2DOrchestrator(bootstrap_if_needed=False)
    yield runtime
    runtime.close()


# Provide a shared MCP client fixture for the test suite
@pytest.fixture(scope="session")
def mcp_client(orchestrator: T2DOrchestrator):
    return orchestrator.mcp_client


# Provide a shared canonical resolver fixture for the test suite
@pytest.fixture(scope="session")
def resolver():
    return get_resolver()
