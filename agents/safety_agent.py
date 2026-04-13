# Imports.
from __future__ import annotations

from core.models import AgentSection
from data.canonical.resolver import CanonicalResolver
from tools.langchain_native_tools import invoke_native_tool
from tools.mcp_client import MCPClientManager
from agents.base_agent import citation, unique_strings


# Safety Agent.
class SafetyAgent:
    """Assemble structured safety, label, and literature context for a drug."""

    def __init__(self, resolver: CanonicalResolver, mcp_client: MCPClientManager) -> None:
        self.resolver = resolver
        self.mcp_client = mcp_client

    def run(self, query: str) -> AgentSection:
        """Return a safety-focused evidence section for the requested therapy."""
        resolved = self.resolver.resolve_drug(query)
        drug_name = (resolved or {}).get("canonical_id", "tirzepatide")
        subgroup = "cardiac_comorbidity_reports" if "cardiac" in query.lower() or "heart" in query.lower() else None
        event_payload = self.mcp_client.call_tool("safety", "search_adverse_events", {"drug": drug_name, "subgroup": subgroup})
        label_payload = self.mcp_client.call_tool("safety", "get_drug_label", {"drug": drug_name})
        summary_payload = self.mcp_client.call_tool("safety", "get_safety_summary", {"drug": drug_name, "subgroup": subgroup})
        literature_payload = invoke_native_tool(
            "search_pubmed_safety_native",
            {"drug": drug_name, "top_k": 2},
        )
        literature = literature_payload["records"]

        top_events = summary_payload["records"][0]["top_events"] if summary_payload["records"] else []
        event_bits = [f"{item['count']} {item['event']} reports" for item in top_events[:2]]
        summary = (
            f"{drug_name.capitalize()} shows reporting patterns including " + ", ".join(event_bits) + "."
            if event_bits
            else f"No structured safety records were found for {drug_name}."
        )
        if subgroup:
            summary += f" The subgroup reviewed was {subgroup.replace('_', ' ')}."

        citations = [
            citation(
                "OpenFDA FAERS",
                f"{drug_name.capitalize()} adverse event signal summary",
                f"FAERS:{drug_name}",
                "https://open.fda.gov/apis/drug/event/",
                "Tier 1",
            )
        ]
        if label_payload["records"]:
            label = label_payload["records"][0]
            citations.append(
                citation(
                    "OpenFDA Drug Label",
                    f"{drug_name.capitalize()} label {label['label_version']}",
                    f"LABEL:{drug_name}",
                    label["source_url"],
                    "Tier 1",
                )
            )
        for document in literature:
            citations.append(
                citation(
                    "PubMed",
                    document["title"],
                    document["pmid"],
                    document["source_url"],
                    "Tier 2",
                    document["publication_date"],
                )
            )

        return AgentSection(
            agent="Safety Agent",
            question_class="Q1",
            summary=summary,
            citations=citations,
            caveats=[
                "Based on FAERS voluntary reporting data.",
                "Reporting patterns do not establish incidence or causality.",
            ],
            evidence_tiers=unique_strings(["Tier 1", "Tier 2"]),
            tool_outputs=[event_payload, label_payload, summary_payload, literature_payload],
            metadata={"drug": drug_name},
        )
