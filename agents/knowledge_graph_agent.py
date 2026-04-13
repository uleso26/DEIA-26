# Imports.
from __future__ import annotations

from core.paths import raw_input_path
from core.models import AgentSection
from core.storage import load_json
from data.canonical.resolver import CanonicalResolver
from tools.langchain_native_tools import invoke_native_tool
from tools.mcp_client import MCPClientManager
from agents.base_agent import citation, dedupe_citations, unique_strings


# Knowledge Graph Agent.
class KnowledgeGraphAgent:
    """Serve both pathway (Q3) and mechanism/graph (Q4) evidence flows."""

    def __init__(self, resolver: CanonicalResolver, mcp_client: MCPClientManager) -> None:
        self.resolver = resolver
        self.mcp_client = mcp_client

    def run(self, query: str, question_class: str = "Q3") -> AgentSection:
        """Dispatch to guideline-pathway or graph-mechanism logic by question class."""
        if question_class == "Q3":
            lowered = query.lower()
            comparison_requested = any(token in lowered for token in ["versus", " vs ", "compare", "differ", "difference"])
            explicit_guideline_comparison = "ada" in lowered and "nice" in lowered
            initial_treatment_requested = self._is_initial_treatment_query(lowered)
            broad_treatment_selection = self._is_broad_treatment_selection_query(lowered)
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
            elif "hypoglyc" in lowered or "avoid lows" in lowered:
                comorbidity = "avoid_hypoglycemia"
            elif "cost" in lowered or "cheap" in lowered or "afford" in lowered:
                comorbidity = "cost_sensitive"
            elif "catabolic" in lowered or "severe hyperglycemia" in lowered or "insulin start" in lowered:
                comorbidity = "severe_hyperglycemia"

            guideline_hint = None
            if "ada" in lowered and "nice" not in lowered:
                guideline_hint = "ADA"
            elif "nice" in lowered and "ada" not in lowered:
                guideline_hint = "NICE"

            phenotype_terms = [
                term
                for term in [
                    "ckd",
                    "obesity",
                    "weight",
                    "heart failure",
                    "ascvd",
                    "hypoglycemia",
                    "cost",
                    "hyperglycemia",
                    "insulin",
                ]
                if term in lowered
            ]
            pathway_payload = self._pathway_payload(
                start_drug=start_drug,
                comorbidity=comorbidity,
                guideline_hint=guideline_hint,
                phenotype_terms=phenotype_terms,
            )
            if explicit_guideline_comparison:
                pathway_payload = self._compare_guideline_payloads(
                    start_drug=start_drug,
                    comorbidity=comorbidity,
                    phenotype_terms=phenotype_terms,
                )
            if initial_treatment_requested and not explicit_guideline_comparison:
                return self._initial_treatment_response(query)
            if broad_treatment_selection and not explicit_guideline_comparison and not comorbidity:
                return self._broad_treatment_selection_response(query)
            context_query = query
            if pathway_payload["records"]:
                context_query = f"{query} {' '.join(record['next_step'] for record in pathway_payload['records'][:2])}"
            clinical_context = invoke_native_tool(
                "get_clinical_context_native",
                {"query": context_query},
            )
            summary = "No pathway logic was found."
            citations = []
            caveats = ["This represents published clinical guideline logic, not medical advice."]
            evidence_tiers = ["Tier 1", "Tier 2"]
            if pathway_payload["records"]:
                if comparison_requested:
                    parts = []
                    distinct_guidelines = unique_strings(
                        [
                            f"{record.get('guideline_name', '')} {record.get('guideline_version', '')}".strip()
                            for record in pathway_payload["records"]
                            if record.get("guideline_name")
                        ]
                    )
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
                    if len(distinct_guidelines) >= 2:
                        summary = f"Guideline comparison after {start_drug}: " + "; ".join(parts) + "."
                    else:
                        summary = (
                            f"A cross-guideline comparison was requested after {start_drug}, but the current bundle "
                            f"returned pathway records from {distinct_guidelines[0] if distinct_guidelines else 'one guideline'} only: "
                            + "; ".join(parts)
                            + "."
                        )
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
                representative = self._select_context_drug(
                    drugbank_records,
                    [record["next_step"] for record in pathway_payload["records"]],
                    start_drug,
                )
                summary += (
                    f" Drug reference context aligned this recommendation with {representative['drug_class']} "
                    f"options such as {representative['canonical_drug']}."
                )
                citations.append(
                    citation(
                        "DrugBank",
                        f"{representative['canonical_drug'].title()} classification",
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
                f"{drug_name.title()} is linked in the graph to targets: {', '.join(targets)}."
                if targets
                else f"No graph targets were found for {drug_name}."
            )
            citations.append(
                citation(
                    "Knowledge Graph",
                    f"{drug_name.title()} target relationships",
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

    def _pathway_payload(
        self,
        *,
        start_drug: str,
        comorbidity: str | None,
        guideline_hint: str | None,
        phenotype_terms: list[str],
    ) -> dict:
        return self.mcp_client.call_tool(
            "knowledge",
            "query_pathway",
            {
                "start_drug": start_drug,
                "comorbidity": comorbidity,
                "guideline_hint": guideline_hint,
                "phenotype_terms": phenotype_terms,
            },
        )

    def _compare_guideline_payloads(
        self,
        *,
        start_drug: str,
        comorbidity: str | None,
        phenotype_terms: list[str],
    ) -> dict:
        guideline_payloads = [
            self._pathway_payload(
                start_drug=start_drug,
                comorbidity=comorbidity,
                guideline_hint=guideline_name,
                phenotype_terms=phenotype_terms,
            )
            for guideline_name in ["ADA", "NICE"]
        ]
        combined_records: list[dict] = []
        for payload in guideline_payloads:
            if payload["records"]:
                combined_records.append(payload["records"][0])
        primary_source = next((payload["source_metadata"] for payload in guideline_payloads if payload["records"]), guideline_payloads[0]["source_metadata"])
        return {
            "tool_name": "query_pathway",
            "canonical_entities": guideline_payloads[0]["canonical_entities"],
            "source_metadata": {
                **primary_source,
                "comparison_guidelines_requested": ["ADA", "NICE"],
            },
            "records": combined_records,
            "raw_provenance": {
                "start_drug": start_drug,
                "comorbidity": comorbidity,
                "guideline_hint": "ADA_vs_NICE",
                "phenotype_terms": phenotype_terms,
            },
        }

    def _select_context_drug(
        self,
        drugbank_records: list[dict[str, str]],
        next_steps: list[str],
        start_drug: str,
    ) -> dict[str, str]:
        next_step_terms = {term.lower() for term in next_steps if term}
        start_drug_lower = start_drug.lower()
        for record in drugbank_records:
            canonical_drug = record.get("canonical_drug", "").lower()
            drug_class = record.get("drug_class", "").lower()
            if canonical_drug in next_step_terms or drug_class in next_step_terms:
                return record
        for record in drugbank_records:
            if record.get("canonical_drug", "").lower() != start_drug_lower:
                return record
        return drugbank_records[0]

    def _is_broad_treatment_selection_query(self, lowered: str) -> bool:
        return (
            any(
                term in lowered
                for term in [
                    "best drug",
                    "best medicine",
                    "best medication",
                    "most effective",
                    "best effective",
                    "strongest",
                    "curing",
                    "cure",
                ]
            )
            or (
                any(term in lowered for term in ["which drug", "which medicine", "which medication"])
                and any(term in lowered for term in ["best", "effective", "cure", "curing", "treatment", "therapy"])
            )
        )

    def _is_initial_treatment_query(self, lowered: str) -> bool:
        return any(
            term in lowered
            for term in [
                "first rx",
                "first-line",
                "first line",
                "initial therapy",
                "initial treatment",
                "newly diagnosed",
                "just diagnosed",
                "starting medicine",
                "starting medication",
                "first prescription",
                "start treatment",
                "start with",
                "first medicine",
                "first medication",
            ]
        )

    def _initial_treatment_response(self, query: str) -> AgentSection:
        guideline_excerpts = load_json(raw_input_path("guideline_excerpts.json"))
        relevant_titles = [
            "ADA Standards of Care 2025: initial pharmacotherapy for newly diagnosed type 2 diabetes",
            "NICE NG28 2026 update: first-line drug treatment at diagnosis",
            "ADA Standards of Care 2025: cardiorenal-first add-on therapy",
            "ADA Standards of Care 2025: insulin initiation for marked hyperglycemia",
        ]
        selected_excerpts = [item for item in guideline_excerpts if item.get("title") in relevant_titles]
        summary = (
            "Usually, metformin is the first-line pharmacotherapy for newly diagnosed Type 2 Diabetes alongside "
            "lifestyle measures when it is tolerated and not contraindicated. Treatment can be adjusted early when "
            "cardiorenal disease, obesity, or drug intolerance materially change priorities, and insulin may be "
            "appropriate if hyperglycaemia is severe or catabolic features are present."
        )
        citations = []
        for item in selected_excerpts:
            source_name = "ADA Standards of Care" if "ADA" in item["title"] else "NICE NG28"
            reference_id = item["id"]
            citations.append(
                citation(
                    source_name,
                    item["title"],
                    reference_id,
                    item.get("source_url") or "https://www.nice.org.uk/guidance/ng28",
                    "Tier 1",
                    item.get("publication_date"),
                )
            )
        return AgentSection(
            agent="Knowledge Graph Agent",
            question_class="Q3",
            summary=summary,
            citations=dedupe_citations(citations),
            caveats=[
                "This reflects general first-line treatment logic, not individualized prescribing advice.",
                "Contraindications, intolerance, cardiorenal disease, obesity priorities, and severity of hyperglycaemia can change the initial regimen.",
            ],
            evidence_tiers=unique_strings(["Tier 1"]),
            tool_outputs=[
                {
                    "tool_name": "guideline_excerpt_bundle",
                    "records": selected_excerpts,
                    "raw_provenance": {"query": query, "response_mode": "initial_treatment_selection"},
                }
            ],
            metadata={
                "guideline_mode": "initial_treatment_selection",
                "force_deterministic_synthesis": True,
            },
        )

    def _broad_treatment_selection_response(self, query: str) -> AgentSection:
        guideline_excerpts = load_json(raw_input_path("guideline_excerpts.json"))
        relevant_titles = [
            "ADA Standards of Care 2025: obesity-aware escalation after metformin",
            "ADA Standards of Care 2025: cardiorenal-first add-on therapy",
            "NICE NG28 2026 update: lower-cost add-on options",
            "ADA Standards of Care 2025: insulin initiation for marked hyperglycemia",
        ]
        selected_excerpts = [item for item in guideline_excerpts if item.get("title") in relevant_titles]
        clinical_context = invoke_native_tool(
            "get_clinical_context_native",
            {"query": query},
        )
        summary = (
            "No single drug cures Type 2 Diabetes, so there is not one universally 'best' treatment. "
            "Guideline-informed selection depends on the main treatment goal: incretin-based therapy such as GLP-1 receptor agonists "
            "or dual GIP/GLP-1 therapy when HbA1c lowering and weight loss are priorities, SGLT2 inhibitors when heart failure or CKD "
            "risk reduction matters, lower-cost options such as sulfonylureas or pioglitazone when affordability is central, and insulin "
            "when hyperglycaemia is severe or rapid control is needed."
        )
        citations = []
        for item in selected_excerpts:
            source_name = "ADA Standards of Care" if "ADA" in item["title"] else "NICE NG28"
            citations.append(
                citation(
                    source_name,
                    item["title"],
                    source_name,
                    item.get("source_url") or "https://www.nice.org.uk/guidance/ng28",
                    "Tier 1",
                )
            )
        return AgentSection(
            agent="Knowledge Graph Agent",
            question_class="Q3",
            summary=summary,
            citations=dedupe_citations(citations),
            caveats=[
                "This reflects general treatment-selection logic, not a claim that any single drug cures Type 2 Diabetes.",
                "The most appropriate therapy depends on glycaemic target, weight goals, CKD/HF/ASCVD profile, hypoglycemia risk, and cost.",
            ],
            evidence_tiers=unique_strings(["Tier 1"]),
            tool_outputs=[
                {
                    "tool_name": "guideline_excerpt_bundle",
                    "records": selected_excerpts,
                },
                clinical_context,
            ],
            metadata={
                "guideline_mode": "broad_treatment_selection",
                "clinical_context_used": bool(clinical_context["records"]),
                "force_deterministic_synthesis": True,
            },
        )
