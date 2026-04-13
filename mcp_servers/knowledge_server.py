# Imports.
from __future__ import annotations

import os
from typing import Any

from core.paths import raw_input_path
from core.storage import load_graph, load_json, run_neo4j_query_with_backend
from data.canonical.resolver import get_resolver
from mcp_servers.base_server import BaseMCPStyleServer


# Knowledge Server.
class KnowledgeServer(BaseMCPStyleServer):
    def __init__(self) -> None:
        super().__init__("knowledge-mcp-server")
        self.resolver = get_resolver()
        self.register_tool("query_pathway", "Return guideline-informed pathway edges for a starting drug and comorbidity.", self.query_pathway)
        self.register_tool("find_drug_targets", "Return graph targets linked to a drug.", self.find_drug_targets)
        self.register_tool("get_mechanism_landscape", "Return a drug/target mechanism landscape from graph and source data.", self.get_mechanism_landscape)

    def _score_pathway_records(
        self,
        records: list[dict[str, Any]],
        comorbidity: str | None,
        guideline_hint: str | None,
        phenotype_terms: list[str],
    ) -> list[dict[str, Any]]:
        scored_records = []
        for index, record in enumerate(records):
            conditions = [condition.lower() for condition in record.get("conditions", [])]
            if comorbidity and comorbidity.lower() not in " ".join(conditions):
                continue
            score = 0
            if comorbidity:
                score += 3
            guideline_text = f"{record.get('guideline_name', '')} {record.get('guideline_version', '')}".lower()
            if guideline_hint:
                if guideline_hint.lower() not in guideline_text:
                    continue
                score += 4
            haystack = " ".join(
                [
                    *record.get("conditions", []),
                    record.get("next_step", ""),
                    record.get("rationale", ""),
                    record.get("guideline_name", ""),
                    record.get("guideline_version", ""),
                ]
            ).lower()
            for term in phenotype_terms:
                if term.lower() in haystack:
                    score += 1
            scored_records.append({**record, "score": score, "ordinal": index})
        scored_records.sort(key=lambda item: (item["score"], -item["ordinal"]), reverse=True)
        return scored_records

    def _pathway_records_from_graph(self, source_node: str) -> list[dict[str, Any]]:
        graph = load_graph()
        guideline_nodes = {node["id"]: node for node in graph["nodes"]}
        records = []
        for edge in graph["edges"]:
            if edge["source"] != source_node or edge["type"] != "NEXT_LINE_IF":
                continue
            target_node = guideline_nodes.get(edge["target"], {"name": edge["target"]})
            guideline = guideline_nodes.get(edge.get("guideline_id", ""), {})
            records.append(
                {
                    "next_step": target_node.get("name"),
                    "conditions": edge.get("conditions", []),
                    "guideline_name": guideline.get("name"),
                    "guideline_version": guideline.get("version"),
                    "guideline_url": guideline.get("url"),
                    "rationale": edge.get("rationale"),
                }
            )
        return records

    def _neo4j_source_metadata(self, backend: str) -> dict[str, Any]:
        if backend == "neo4j":
            return {
                "storage": "neo4j",
                "database": os.getenv("NEO4J_DATABASE", "neo4j"),
            }
        return {
            "storage": "neo4j_fallback",
            "graph_file": "data/processed/neo4j_graph.json",
        }

    def query_pathway(
        self,
        start_drug: str,
        comorbidity: str | None = None,
        guideline_hint: str | None = None,
        phenotype_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved = self.resolver.resolve_drug(start_drug) or {"canonical_id": start_drug.lower()}
        source_node = f"drug:{resolved['canonical_id']}"
        phenotype_terms = phenotype_terms or []
        live_records, backend = run_neo4j_query_with_backend(
            """
            MATCH (source:Entity {id: $source_node})-[r:RELATIONSHIP {relation_type: 'NEXT_LINE_IF'}]->(target:Entity)
            OPTIONAL MATCH (guideline:Entity {id: r.guideline_id})
            RETURN target.name AS next_step,
                   coalesce(r.conditions, []) AS conditions,
                   guideline.name AS guideline_name,
                   guideline.version AS guideline_version,
                   guideline.url AS guideline_url,
                   r.rationale AS rationale
            """,
            {"source_node": source_node},
        )
        records = self._score_pathway_records(live_records, comorbidity, guideline_hint, phenotype_terms)
        source_metadata = self._neo4j_source_metadata(backend)
        if not records:
            fallback_records = self._score_pathway_records(
                self._pathway_records_from_graph(source_node),
                comorbidity,
                guideline_hint,
                phenotype_terms,
            )
            if fallback_records:
                records = fallback_records
                source_metadata = {
                    "storage": "neo4j_fallback",
                    "graph_file": "data/processed/neo4j_graph.json",
                    "fallback_reason": "live_query_returned_no_pathway_records",
                }
        return {
            "tool_name": "query_pathway",
            "canonical_entities": {"drug": resolved},
            "source_metadata": source_metadata,
            "records": records,
            "raw_provenance": {
                "start_drug": start_drug,
                "comorbidity": comorbidity,
                "guideline_hint": guideline_hint,
                "phenotype_terms": phenotype_terms,
            },
        }

    def find_drug_targets(self, drug: str) -> dict[str, Any]:
        resolved = self.resolver.resolve_drug(drug) or {"canonical_id": drug.lower()}
        source_node = f"drug:{resolved['canonical_id']}"
        graph = load_graph()
        nodes = {node["id"]: node for node in graph["nodes"]}

        def graph_fallback_records() -> list[dict[str, Any]]:
            fallback_records = []
            for edge in graph["edges"]:
                if edge["source"] == source_node and edge["type"] == "TARGETS":
                    target = nodes.get(edge["target"], {})
                    fallback_records.append(
                        {
                            "target": target.get("name"),
                            "target_id": edge["target"],
                            "source_dataset": edge.get("source_dataset"),
                            "validated": edge.get("validated"),
                            "evidence_source": edge.get("evidence_source"),
                        }
                    )
            return fallback_records

        live_records, backend = run_neo4j_query_with_backend(
            """
            MATCH (source:Entity {id: $source_node})-[r:RELATIONSHIP {relation_type: 'TARGETS'}]->(target:Entity)
            RETURN target.name AS target,
                   target.id AS target_id,
                   r.source_dataset AS source_dataset,
                   r.validated AS validated,
                   r.evidence_source AS evidence_source
            """,
            {"source_node": source_node},
        )
        if backend == "neo4j":
            records = live_records
            source_metadata = self._neo4j_source_metadata(backend)
        else:
            records = graph_fallback_records()
            source_metadata = self._neo4j_source_metadata(backend)
        if not records:
            fallback_records = graph_fallback_records()
            if fallback_records:
                records = fallback_records
                source_metadata = {
                    "storage": "neo4j_fallback",
                    "graph_file": "data/processed/neo4j_graph.json",
                    "fallback_reason": "live_query_returned_no_target_records",
                }
        return {
            "tool_name": "find_drug_targets",
            "canonical_entities": {"drug": resolved},
            "source_metadata": source_metadata,
            "records": records,
            "raw_provenance": {"drug": drug},
        }

    def get_mechanism_landscape(self, drug: str | None = None, target: str | None = None) -> dict[str, Any]:
        resolved_drug = self.resolver.resolve_drug(drug) if drug else None
        resolved_target = self.resolver.resolve_target(target) if target else None
        chembl = load_json(raw_input_path("chembl.json"))
        opentargets = load_json(raw_input_path("opentargets.json"))
        uniprot = load_json(raw_input_path("uniprot.json"))

        records = []
        datasets = ["chembl", "opentargets", "uniprot"]
        if resolved_drug:
            records.extend([item for item in chembl if item["canonical_drug"] == resolved_drug["canonical_id"]])
            records.extend([item for item in opentargets if item["canonical_drug"] == resolved_drug["canonical_id"]])
            graph_records, backend = run_neo4j_query_with_backend(
                """
                MATCH (drug:Entity {id: $drug_id})-[r:RELATIONSHIP {relation_type: 'TARGETS'}]->(target:Entity)
                RETURN drug.name AS canonical_drug,
                       target.name AS canonical_target,
                       r.source_dataset AS source_dataset,
                       r.validated AS validated,
                       r.evidence_source AS evidence_source
                """,
                {"drug_id": f"drug:{resolved_drug['canonical_id']}"},
            )
            if backend == "neo4j":
                records.extend(graph_records)
                datasets.append("neo4j")
        if resolved_target:
            records.extend([item for item in uniprot if item["canonical_target"] == resolved_target["canonical_id"]])
            records.extend([item for item in opentargets if item["canonical_target"] == resolved_target["canonical_id"]])
            graph_records, backend = run_neo4j_query_with_backend(
                """
                MATCH (drug:Entity)-[r:RELATIONSHIP {relation_type: 'TARGETS'}]->(target:Entity {id: $target_id})
                RETURN drug.name AS canonical_drug,
                       target.name AS canonical_target,
                       r.source_dataset AS source_dataset,
                       r.validated AS validated,
                       r.evidence_source AS evidence_source
                """,
                {"target_id": f"target:{resolved_target['canonical_id']}"},
            )
            if backend == "neo4j":
                records.extend(graph_records)
                datasets.append("neo4j")
        return {
            "tool_name": "get_mechanism_landscape",
            "canonical_entities": {"drug": resolved_drug, "target": resolved_target},
            "source_metadata": {"storage": "mixed_sources", "datasets": sorted(set(datasets))},
            "records": records,
            "raw_provenance": {"drug": drug, "target": target},
        }


# Main.
def main() -> None:
    KnowledgeServer().run_cli()


# CLI entrypoint.
if __name__ == "__main__":
    main()
