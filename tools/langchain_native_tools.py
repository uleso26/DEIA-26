# Imports.
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from core.storage import load_retrieval_manifest
from tools.context_tools import (
    fetch_trial_results,
    get_clinical_context,
    get_population_context,
)
from tools.retrieval import (
    search_pubmed_safety,
    search_retrieval_index,
)


# Search retrieval index native.
@tool("search_retrieval_index_native")
def search_retrieval_index_native(query: str, top_k: int = 4) -> dict[str, Any]:
    """Search the hybrid literature retrieval layer for evidence relevant to a query."""
    manifest = load_retrieval_manifest()
    return {
        "tool_name": "search_retrieval_index_native",
        "source_metadata": {
            "storage": "retrieval_manifest",
            "backend": manifest.get("backend", "lexical"),
        },
        "records": search_retrieval_index(query, top_k=top_k),
        "raw_provenance": {
            "query": query,
            "top_k": top_k,
        },
    }


# Get population context native.
@tool("get_population_context_native")
def get_population_context_native(country: str | None = None, top_k: int = 2) -> dict[str, Any]:
    """Return WHO population surveillance context for a country or population query."""
    return get_population_context(country=country, top_k=top_k)


# Get clinical context native.
@tool("get_clinical_context_native")
def get_clinical_context_native(query: str) -> dict[str, Any]:
    """Return DrugBank and synthetic patient context matched to the query."""
    return get_clinical_context(query)


# Fetch trial results native.
@tool("fetch_trial_results_native")
def fetch_trial_results_native(query: str) -> dict[str, Any]:
    """Return curated trial-result records matched to the query."""
    return {
        "tool_name": "fetch_trial_results_native",
        "source_metadata": {
            "storage": "mongo_fallback",
            "collection": "clinical_trials",
        },
        "records": fetch_trial_results(query),
        "raw_provenance": {
            "query": query,
        },
    }


# Search PubMed safety native.
@tool("search_pubmed_safety_native")
def search_pubmed_safety_native(drug: str, top_k: int = 2) -> dict[str, Any]:
    """Return safety-oriented PubMed matches for a drug query."""
    return {
        "tool_name": "search_pubmed_safety_native",
        "source_metadata": {
            "storage": "retrieval_manifest",
            "dataset": "pubmed_documents",
        },
        "records": search_pubmed_safety(drug, top_k=top_k),
        "raw_provenance": {
            "drug": drug,
            "top_k": top_k,
        },
    }


# Module constants.
NATIVE_LANGCHAIN_TOOLS: dict[str, BaseTool] = {
    item.name: item
    for item in [
        search_retrieval_index_native,
        get_population_context_native,
        get_clinical_context_native,
        fetch_trial_results_native,
        search_pubmed_safety_native,
    ]
}


# Invoke native tool.
def invoke_native_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Invoke a native LangChain tool and normalize the result payload."""
    if name not in NATIVE_LANGCHAIN_TOOLS:
        raise KeyError(f"Unknown native LangChain tool: {name}")
    payload = NATIVE_LANGCHAIN_TOOLS[name].invoke(arguments)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Native LangChain tool {name} returned a non-dict payload.")
    return payload


# Describe native tools.
def describe_native_tools(tool_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Describe the selected native LangChain tools for planner prompts."""
    selected_names = tool_names or list(NATIVE_LANGCHAIN_TOOLS)
    described: list[dict[str, Any]] = []
    for name in selected_names:
        tool_obj = NATIVE_LANGCHAIN_TOOLS[name]
        described.append(
            {
                "name": tool_obj.name,
                "description": tool_obj.description,
                "args": getattr(tool_obj, "args", {}),
            }
        )
    return described
