from __future__ import annotations

from core.models import AgentSection
from data.canonical.resolver import CanonicalResolver
from tools.mcp_client import MCPClientManager
from tools.native_tools import get_clinical_context
from agents.base_agent import citation, unique_strings


class KnowledgeGraphAgent:
    def __init__(self, resolver: CanonicalResolver, mcp_client: MCPClientManager) -> None:
        self.resolver = resolver
        self.mcp_client = mcp_client

    def run(self, query: str, question_class: str = "Q3") -> AgentSection:
        if question_class == "Q3":
            lowered = query.lower()
            start_drug = "metformin" if "metformin" in query.lower() else (self.resolver.resolve_drug(query) or {}).get("canonical_id", "metformin")
            comorbidity = None
            if "ckd" in lowered:
                comorbidity = "CKD stage 3"
            elif "obesity" in lowered or "weight" in lowered:
                comorbidity = "obesity"
            elif "heart failure" in lowered or "hf" in lowered:
                comorbidity = "heart failure"
            elif "ascvd" in lowered or "atherosclerotic" in lowered:
                comorbidity = "ASCVD"

            guideline_hint = None
            if "ada" in lowered and "nice" not in lowered:
                guideline_hint = "ADA"
            elif "nice" in lowered and "ada" not in lowered:
                guideline_hint = "NICE"

            phenotype_terms = [term for term in ["ckd", "obesity", "weight", "heart failure", "cvd"] if term in lowered]
            pathway_payload = self.mcp_client.call_tool(
                "knowledge",
                "query_pathway",
                {
                    "start_drug": start_drug,
                    "comorbidity": comorbidity,
                    "guideline_hint": guideline_hint,
                    "phenotype_terms": phenotype_terms,
                },
            )
            context_query = query
            if pathway_payload["records"]:
                context_query = f"{query} {' '.join(record['next_step'] for record in pathway_payload['records'][:2])}"
            clinical_context = get_clinical_context(context_query)
            summary = "No pathway logic was found."
            citations = []
            caveats = ["This represents published clinical guideline logic, not medical advice."]
            evidence_tiers = ["Tier 1", "Tier 2"]
            if pathway_payload["records"]:
                if any(token in lowered for token in ["versus", " vs ", "compare"]):
                    parts = []
                    for record in pathway_payload["records"][:2]:
                        parts.append(
                            f"{record['guideline_name']} {record['guideline_version']} prioritizes {record['next_step']}"
                        )
                        citations.append(
                            citation(
                                record["guideline_name"],
                                f"{record['guideline_name']} {record['guideline_version']}",
                                record["guideline_name"],
                                record.get("guideline_url") or "https://www.nice.org.uk/guidance/ng28",
                                "Tier 1",
                            )
                        )
                    summary = f"Guideline comparison after {start_drug}: " + "; ".join(parts) + "."
                else:
                    record = pathway_payload["records"][0]
                    summary = (
                        f"{record['guideline_name']} {record['guideline_version']} pathway logic prioritizes "
                        f"{record['next_step']} after {start_drug} for patients matching {', '.join(record['conditions'])}."
                    )
                    citations.append(
                        citation(
                            record["guideline_name"],
                            f"{record['guideline_name']} {record['guideline_version']}",
                            record["guideline_name"],
                            record.get("guideline_url") or "https://www.nice.org.uk/guidance/ng28",
                            "Tier 1",
                        )
                    )
            drugbank_records = [record for record in clinical_context["records"] if record["record_type"] == "drugbank"]
            synthetic_records = [record for record in clinical_context["records"] if record["record_type"] == "synthetic_profile"]
            if drugbank_records:
                representative = drugbank_records[0]
                summary += (
                    f" Drug reference context aligned this recommendation with {representative['drug_class']} "
                    f"options such as {representative['canonical_drug']}."
                )
                citations.append(
                    citation(
                        "DrugBank",
                        f"{representative['canonical_drug'].capitalize()} classification",
                        representative["drugbank_id"],
                        representative["source_url"],
                        "Tier 2",
                    )
                )
            if synthetic_records:
                representative = synthetic_records[0]
                summary += (
                    f" A matched synthetic cohort profile reflected {representative['bmi_category']} status, "
                    f"{representative['ckd_stage']}, and {representative['current_therapy']} background."
                )
                caveats.append("Synthetic patient profiles are illustrative internal context, not real-world evidence.")
            return AgentSection(
                agent="Knowledge Graph Agent",
                question_class="Q3",
                summary=summary,
                citations=citations,
                caveats=unique_strings(caveats),
                evidence_tiers=unique_strings(evidence_tiers),
                tool_outputs=[pathway_payload, clinical_context],
                metadata={
                    "guideline_mode": "guideline-informed",
                    "clinical_context_used": bool(clinical_context["records"]),
                },
            )

        resolved = self.resolver.resolve_drug(query)
        resolved_target = self.resolver.resolve_target(query)
        citations = []
        tool_outputs = []

        if resolved:
            drug_name = resolved["canonical_id"]
            target_payload = self.mcp_client.call_tool("knowledge", "find_drug_targets", {"drug": drug_name})
            tool_outputs.append(target_payload)
            targets = [record["target"] for record in target_payload["records"]]
            summary = (
                f"{drug_name.capitalize()} is linked in the graph to targets: {', '.join(targets)}."
                if targets
                else f"No graph targets were found for {drug_name}."
            )
            citations.append(
                citation(
                    "Knowledge Graph",
                    f"{drug_name.capitalize()} target relationships",
                    f"GRAPH:{drug_name}",
                    "data/processed/neo4j_graph.json",
                    "Tier 1",
                )
            )
            metadata = {"drug": drug_name}
        elif resolved_target:
            target_name = resolved_target["canonical_id"]
            landscape_payload = self.mcp_client.call_tool(
                "knowledge",
                "get_mechanism_landscape",
                {"target": target_name},
            )
            tool_outputs.append(landscape_payload)
            linked_drugs = sorted({record["canonical_drug"] for record in landscape_payload["records"] if record.get("canonical_drug")})
            summary = (
                f"{target_name} is linked to drugs including {', '.join(linked_drugs)}."
                if linked_drugs
                else f"No drug relationships were found for target {target_name}."
            )
            citations.append(
                citation(
                    "OpenTargets",
                    f"{target_name} target landscape",
                    f"OPENTARGETS:{target_name}",
                    "https://platform.opentargets.org/",
                    "Tier 2",
                )
            )
            metadata = {"target": target_name}
        else:
            summary = "No drug or target entities were resolved from the query."
            metadata = {}
        return AgentSection(
            agent="Knowledge Graph Agent",
            question_class="Q4",
            summary=summary,
            citations=citations,
            caveats=["Mechanism statements should distinguish validated evidence from inferred associations."],
            evidence_tiers=unique_strings(["Tier 1", "Tier 2"]),
            tool_outputs=tool_outputs,
            metadata=metadata,
        )
