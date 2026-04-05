from __future__ import annotations

from unittest.mock import patch

import pytest


pytestmark = pytest.mark.integration


def test_safety_mcp_tool_returns_records(mcp_client) -> None:
    result = mcp_client.call_tool("safety", "search_adverse_events", {"drug": "tirzepatide"})
    assert result["canonical_entities"]["drug"]["canonical_id"] == "tirzepatide"
    assert len(result["records"]) >= 1
    assert result["server_name"] == "safety"
    assert "requested_at" in result


def test_mcp_list_tools_exposes_structured_schemas(mcp_client) -> None:
    tools = mcp_client.list_tools("safety")
    assert len(tools) >= 1
    search_tool = next(tool for tool in tools if tool["name"] == "search_adverse_events")
    assert "inputSchema" in search_tool
    assert "outputSchema" in search_tool


def test_knowledge_server_falls_back_when_live_neo4j_returns_empty() -> None:
    from mcp_servers.knowledge_server import KnowledgeServer

    server = KnowledgeServer()
    with patch("mcp_servers.knowledge_server.run_neo4j_query_with_backend", return_value=([], "neo4j")):
        payload = server.query_pathway(
            start_drug="metformin",
            comorbidity="obesity",
            guideline_hint="ADA",
            phenotype_terms=["obesity"],
        )
    assert len(payload["records"]) >= 1
    assert payload["source_metadata"]["storage"] == "neo4j_fallback"


def test_unknown_drug_safety_tool_returns_empty_records(mcp_client) -> None:
    result = mcp_client.call_tool("safety", "search_adverse_events", {"drug": "unknowncompound"})
    assert result["records"] == []
