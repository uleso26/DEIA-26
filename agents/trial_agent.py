# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from core.models import AgentSection
from data.canonical.resolver import CanonicalResolver
from tools.langchain_native_tools import invoke_native_tool
from tools.mcp_client import MCPClientManager
from agents.base_agent import citation, unique_strings


# Define the trial agent and its specialist response logic
class TrialAgent:
    """Handle trial-detail and efficacy-comparison questions."""

    def __init__(self, resolver: CanonicalResolver, mcp_client: MCPClientManager) -> None:
        self.resolver = resolver
        self.mcp_client = mcp_client

    def run(self, query: str) -> AgentSection:
        """Return a grounded trial section for a trial name or drug comparison query."""
        trial_match = self.resolver.resolve_trial(query)
        drugs = self.resolver.find_drugs(query)
        tool_outputs = []
        if trial_match:
            compare_payload = self.mcp_client.call_tool(
                "trials",
                "get_trial_detail",
                {"trial_id": trial_match["canonical_id"]},
            )
            tool_outputs.append(compare_payload)
        elif len(drugs) >= 2:
            compare_payload = self.mcp_client.call_tool(
                "trials",
                "compare_trials",
                {"drug_a": drugs[0]["canonical_id"], "drug_b": drugs[1]["canonical_id"]},
            )
            tool_outputs.append(compare_payload)
        else:
            canonical = drugs[0]["canonical_id"] if drugs else "semaglutide"
            compare_payload = self.mcp_client.call_tool("trials", "search_trials", {"drug": canonical, "phase": "Phase 3"})
            tool_outputs.append(compare_payload)

        native_results_payload = invoke_native_tool("fetch_trial_results_native", {"query": query})
        native_results = native_results_payload["records"]
        records = compare_payload["records"] or native_results
        comparison_type = None
        if records and len(records) == 1:
            comparison_type = records[0].get("comparison_type", "indirect")
        elif records:
            comparison_type = compare_payload["raw_provenance"].get("comparison_type", "indirect")

        summary = "No trial comparison data was found."
        citations = []
        if records:
            primary = records[0]
            if primary["trial_name"] == "SURPASS-2" and len(records) == 1:
                results = primary["results"]
                summary = (
                    "SURPASS-2 provides direct phase 3 comparison evidence: tirzepatide achieved HbA1c reductions of "
                    f"{results['tirzepatide_10mg']['hba1c_change']} to {results['tirzepatide_15mg']['hba1c_change']} "
                    f"versus {results['semaglutide_1mg']['hba1c_change']} for semaglutide 1 mg at week 40."
                )
            elif len(records) == 1:
                arm_summaries = []
                for arm_name, result in list(primary["results"].items())[:2]:
                    metric_fragments = []
                    if "hba1c_change" in result:
                        metric_fragments.append(f"HbA1c change {result['hba1c_change']}")
                    if "weight_change_kg" in result:
                        metric_fragments.append(f"weight change {result['weight_change_kg']} kg")
                    if "mace_hazard_ratio" in result:
                        metric_fragments.append(f"MACE hazard ratio {result['mace_hazard_ratio']}")
                    if "cv_death_hazard_ratio" in result:
                        metric_fragments.append(f"cardiovascular death hazard ratio {result['cv_death_hazard_ratio']}")
                    if "primary_endpoint_hazard_ratio" in result:
                        metric_fragments.append(f"primary endpoint hazard ratio {result['primary_endpoint_hazard_ratio']}")
                    if not metric_fragments:
                        metric_fragments.append("reported endpoint summary available")
                    arm_summaries.append(f"{arm_name} " + "; ".join(metric_fragments))
                summary = (
                    f"{primary['trial_name']} is a completed {primary['phase']} study with "
                    f"{comparison_type} evidence around {primary['primary_endpoint']}. "
                    f"Key reported results include {'; '.join(arm_summaries)}."
                )
            else:
                trial_names = ", ".join(record["trial_name"] for record in records[:3])
                summary = (
                    f"Retrieved {comparison_type} phase 3 comparison evidence across trials including {trial_names}."
                )
            for record in records:
                citations.append(
                    citation(
                        "ClinicalTrials.gov",
                        record["trial_name"],
                        record["nct_id"],
                        record["source_url"],
                        "Tier 2",
                        record["completion_date"],
                    )
                )
                publication_reference = record.get("publication_reference")
                if publication_reference:
                    citations.append(
                        citation(
                            "PubMed",
                            f"{record['trial_name']} publication",
                            publication_reference,
                            f"https://pubmed.ncbi.nlm.nih.gov/{publication_reference.replace('PMID', '')}/",
                            "Tier 2",
                        )
                    )

        return AgentSection(
            agent="Clinical Trial Agent",
            question_class="Q2",
            summary=summary,
            citations=citations,
            caveats=["Comparisons should distinguish direct head-to-head evidence from indirect cross-trial inference."],
            evidence_tiers=unique_strings(["Tier 2"]),
            tool_outputs=[*tool_outputs, native_results_payload],
            metadata={"comparison_type": comparison_type, "trial_id": (trial_match or {}).get("canonical_id")},
        )
