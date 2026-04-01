from __future__ import annotations

from functools import lru_cache

from core.models import AgentSection
from core.paths import RAW_DIR
from core.storage import load_json
from data.canonical.resolver import CanonicalResolver
from tools.mcp_client import MCPClientManager
from tools.native_tools import query_chembl, query_uniprot
from agents.base_agent import citation, unique_strings


@lru_cache(maxsize=1)
def _load_opentargets() -> list[dict]:
    return load_json(RAW_DIR / "opentargets.json")


class MoleculeAgent:
    def __init__(self, resolver: CanonicalResolver, mcp_client: MCPClientManager) -> None:
        self.resolver = resolver
        self.mcp_client = mcp_client

    def run(self, query: str) -> AgentSection:
        resolved_drug = self.resolver.resolve_drug(query)
        resolved_target = self.resolver.resolve_target(query)
        drug_name = (resolved_drug or {}).get("canonical_id")
        target_name = (resolved_target or {}).get("canonical_id")

        landscape_payload = self.mcp_client.call_tool(
            "knowledge",
            "get_mechanism_landscape",
            {"drug": drug_name, "target": target_name},
        )
        chembl_records = query_chembl(drug_name) if drug_name else []
        uniprot_records = query_uniprot(target_name) if target_name else []
        targets = sorted({item.get("canonical_target") for item in landscape_payload["records"] if item.get("canonical_target")})
        opentargets = _load_opentargets()
        mechanism = chembl_records[0]["mechanism"] if chembl_records else "No mechanism metadata found"
        linked_drugs = sorted({item.get("canonical_drug") for item in landscape_payload["records"] if item.get("canonical_drug")})
        shared_drugs = []
        if drug_name and targets:
            shared_drugs = sorted(
                {
                    item["canonical_drug"]
                    for item in opentargets
                    if item["canonical_target"] in targets and item["canonical_drug"] != drug_name
                }
            )

        if drug_name:
            summary = f"{drug_name.capitalize()} is described as {mechanism}."
            if targets:
                summary += f" Shared or linked targets include {', '.join(targets)}."
            if shared_drugs:
                summary += f" Other drugs sharing part of this mechanism include {', '.join(shared_drugs)}."
        elif target_name:
            summary = f"{target_name} has {len(uniprot_records)} curated protein annotation record(s)."
            if uniprot_records:
                summary += f" UniProt describes it as {uniprot_records[0]['protein_name']}."
        else:
            summary = "No molecule or target entities were resolved from the query."

        citations = []
        for record in chembl_records:
            citations.append(
                citation(
                    "ChEMBL",
                    f"{record['canonical_drug']} mechanism",
                    record["chembl_id"],
                    "https://www.ebi.ac.uk/chembl/",
                    "Tier 2",
                )
            )
        for record in uniprot_records:
            citations.append(
                citation(
                    "UniProt",
                    record["protein_name"],
                    record["uniprot_id"],
                    "https://www.uniprot.org/",
                    "Tier 2",
                )
            )
        if drug_name:
            citations.append(
                citation(
                    "OpenTargets",
                    f"{drug_name} target landscape",
                    f"OPENTARGETS:{drug_name}",
                    "https://platform.opentargets.org/",
                    "Tier 2",
                )
            )
        if target_name:
            citations.append(
                citation(
                    "OpenTargets",
                    f"{target_name} associated drugs",
                    f"OPENTARGETS:{target_name}",
                    "https://platform.opentargets.org/",
                    "Tier 2",
                )
            )
        return AgentSection(
            agent="Molecule Agent",
            question_class="Q4",
            summary=summary,
            citations=citations,
            caveats=["Mechanism statements should distinguish validated evidence from inferred associations."],
            evidence_tiers=unique_strings(["Tier 2"]),
            tool_outputs=[landscape_payload],
            metadata={"drug": drug_name, "target": target_name},
        )
