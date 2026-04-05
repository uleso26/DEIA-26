from __future__ import annotations

import pytest

from agents.orchestrator import T2DOrchestrator, bootstrap_runtime
from data.canonical.resolver import get_resolver


@pytest.fixture(scope="session")
def bootstrap_data() -> None:
    bootstrap_runtime()


@pytest.fixture(scope="session")
def orchestrator(bootstrap_data: None) -> T2DOrchestrator:
    runtime = T2DOrchestrator(bootstrap_if_needed=False)
    yield runtime
    runtime.close()


@pytest.fixture(scope="session")
def mcp_client(orchestrator: T2DOrchestrator):
    return orchestrator.mcp_client


@pytest.fixture(scope="session")
def resolver():
    return get_resolver()
