from __future__ import annotations

from typing import Any

from data.canonical.resolver import get_resolver
from mcp_servers.base_server import BaseMCPStyleServer
from core.storage import load_collection_with_backend


class TrialsServer(BaseMCPStyleServer):
    def __init__(self) -> None:
        super().__init__("trials-mcp-server")
        self.resolver = get_resolver()
        self.register_tool("search_trials", "Search trial documents by drug, phase, or condition.", self.search_trials)
        self.register_tool("get_trial_detail", "Return the detail for a specific NCT identifier or trial name.", self.get_trial_detail)
        self.register_tool("compare_trials", "Compare trial efficacy results for one or more drugs.", self.compare_trials)

    def _trial_documents(self) -> tuple[list[dict[str, Any]], str]:
        return load_collection_with_backend("clinical_trials")

    def search_trials(
        self,
        drug: str | None = None,
        phase: str | None = None,
        condition: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        documents, backend = self._trial_documents()
        resolved = self.resolver.resolve_drug(drug) if drug else None
        matches = []
        for document in documents:
            if resolved and resolved["canonical_id"] not in [item.lower() for item in document["interventions"]]:
                continue
            if phase and phase.lower() not in document["phase"].lower():
                continue
            if condition and not any(condition.lower() in item.lower() for item in document["conditions"]):
                continue
            matches.append(document)
        return {
            "tool_name": "search_trials",
            "canonical_entities": {"drug": resolved},
            "source_metadata": {"storage": backend, "collection": "clinical_trials"},
            "records": matches[:top_k],
            "raw_provenance": {"query_drug": drug, "phase": phase, "condition": condition},
        }

    def get_trial_detail(self, trial_id: str) -> dict[str, Any]:
        trial_match = self.resolver.resolve_trial(trial_id)
        documents, backend = self._trial_documents()
        normalized = (trial_match or {}).get("canonical_id", trial_id).lower()
        records = [
            item
            for item in documents
            if item["nct_id"].lower() == normalized or item["trial_name"].lower() == trial_id.lower()
        ]
        return {
            "tool_name": "get_trial_detail",
            "canonical_entities": {"trial": trial_match},
            "source_metadata": {"storage": backend, "collection": "clinical_trials"},
            "records": records,
            "raw_provenance": {"trial_id": trial_id},
        }

    def compare_trials(self, drug_a: str, drug_b: str | None = None) -> dict[str, Any]:
        documents, backend = self._trial_documents()
        resolved_a = self.resolver.resolve_drug(drug_a) or {"canonical_id": drug_a.lower()}
        resolved_b = self.resolver.resolve_drug(drug_b) if drug_b else None

        candidate_records = []
        for document in documents:
            interventions = [item.lower() for item in document["interventions"]]
            if resolved_a["canonical_id"] not in interventions:
                continue
            if resolved_b and resolved_b["canonical_id"] not in interventions:
                continue
            candidate_records.append(document)

        if not candidate_records and resolved_b:
            for document in documents:
                interventions = [item.lower() for item in document["interventions"]]
                if resolved_a["canonical_id"] in interventions or resolved_b["canonical_id"] in interventions:
                    candidate_records.append(document)

        comparison_type = "direct" if any(record["comparison_type"] == "direct" for record in candidate_records) else "indirect"
        return {
            "tool_name": "compare_trials",
            "canonical_entities": {"drug_a": resolved_a, "drug_b": resolved_b},
            "source_metadata": {"storage": backend, "collection": "clinical_trials"},
            "records": candidate_records,
            "raw_provenance": {
                "drug_a": drug_a,
                "drug_b": drug_b,
                "comparison_type": comparison_type,
            },
        }


def main() -> None:
    TrialsServer().run_cli()


if __name__ == "__main__":
    main()
