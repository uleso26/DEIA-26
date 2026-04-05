from __future__ import annotations

import pytest

from tools.context_tools import get_clinical_context, get_population_context
from tools.langchain_native_tools import invoke_native_tool


pytestmark = pytest.mark.integration


def test_population_context_tool_returns_who_record() -> None:
    payload = get_population_context("United Kingdom")
    assert len(payload["records"]) >= 1
    assert payload["records"][0]["country"] == "United Kingdom"


def test_langchain_native_tool_returns_structured_payload() -> None:
    payload = invoke_native_tool(
        "get_population_context_native",
        {"country": "United Kingdom", "top_k": 1},
    )
    assert payload["tool_name"] == "get_population_context"
    assert payload["records"][0]["country"] == "United Kingdom"


def test_clinical_context_tool_returns_drugbank_and_synthetic_matches() -> None:
    payload = get_clinical_context("ADA pathway after metformin for obesity GLP-1 receptor agonist")
    record_types = {record["record_type"] for record in payload["records"]}
    assert "drugbank" in record_types
    assert "synthetic_profile" in record_types
