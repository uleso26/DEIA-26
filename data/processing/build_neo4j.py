from __future__ import annotations

import argparse
import os
import time
from typing import Any

from core.paths import GRAPH_FILE, RAW_DIR
from core.storage import dump_json, load_json
from data.ingestion.seed_data import GUIDELINE_GRAPH


def _clean_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in properties.items() if value is not None}


def _clear_graph(tx: Any) -> None:
    tx.run("MATCH (n) DETACH DELETE n").consume()


def _write_graph(tx: Any, graph: dict[str, Any]) -> None:
    for node in graph["nodes"]:
        tx.run(
            "CREATE (n:Entity) SET n += $properties",
            properties=_clean_properties(
                {
                    "id": node["id"],
                    "type": node["type"],
                    "name": node["name"],
                    "version": node.get("version"),
                    "url": node.get("url"),
                }
            ),
        ).consume()
    for edge in graph["edges"]:
        tx.run(
            """
            MATCH (source:Entity {id: $source}), (target:Entity {id: $target})
            CREATE (source)-[r:RELATIONSHIP]->(target)
            SET r += $properties
            """,
            source=edge["source"],
            target=edge["target"],
            properties=_clean_properties(
                {
                    "relation_type": edge["type"],
                    "conditions": edge.get("conditions", []),
                    "guideline_id": edge.get("guideline_id"),
                    "rationale": edge.get("rationale"),
                    "source_dataset": edge.get("source_dataset"),
                    "validated": edge.get("validated"),
                    "evidence_source": edge.get("evidence_source"),
                }
            ),
        ).consume()


def _sync_to_neo4j(graph: dict) -> None:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not all([uri, user, password]):
        return
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        return

    retries = int(os.getenv("NEO4J_SYNC_RETRIES", "10"))
    delay_seconds = float(os.getenv("NEO4J_SYNC_DELAY_SECONDS", "1.5"))
    last_error: Exception | None = None

    for _ in range(retries):
        driver = None
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                session.execute_write(_clear_graph)
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE").consume()
                session.execute_write(_write_graph, graph)
                entity_count = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
                relation_count = session.run("MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) AS count").single()["count"]
            if entity_count < len(graph["nodes"]) or relation_count < len(graph["edges"]):
                raise RuntimeError(
                    f"Neo4j graph sync was incomplete: entities={entity_count}/{len(graph['nodes'])}, "
                    f"relations={relation_count}/{len(graph['edges'])}"
                )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(delay_seconds)
        finally:
            if driver is not None:
                driver.close()
    if last_error is not None:
        raise RuntimeError(f"Neo4j sync failed after {retries} attempts: {last_error}") from last_error


def run(sync: bool = False) -> str:
    graph = {
        "nodes": [dict(node) for node in GUIDELINE_GRAPH["nodes"]],
        "edges": [dict(edge) for edge in GUIDELINE_GRAPH["edges"]],
    }
    if (RAW_DIR / "opentargets.json").exists():
        opentargets = load_json(RAW_DIR / "opentargets.json")
        node_ids = {node["id"] for node in graph["nodes"]}
        edge_keys = {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]}
        for item in opentargets:
            drug_node = f"drug:{item['canonical_drug']}"
            target_node = f"target:{item['canonical_target']}"
            disease_slug = item.get("canonical_disease", "Type 2 Diabetes").lower().replace(" ", "_")
            disease_node = f"disease:{disease_slug}"
            if disease_node not in node_ids:
                graph["nodes"].append(
                    {"id": disease_node, "type": "Disease", "name": item.get("canonical_disease", "Type 2 Diabetes")}
                )
                node_ids.add(disease_node)
            edge_key = (drug_node, target_node, "TARGETS")
            if edge_key not in edge_keys:
                graph["edges"].append(
                    {
                        "source": drug_node,
                        "target": target_node,
                        "type": "TARGETS",
                        "source_dataset": "OpenTargets",
                        "validated": item.get("validated", False),
                        "evidence_source": item.get("evidence_source"),
                    }
                )
                edge_keys.add(edge_key)
            disease_edge_key = (drug_node, disease_node, "TREATS")
            if disease_edge_key not in edge_keys:
                graph["edges"].append(
                    {
                        "source": drug_node,
                        "target": disease_node,
                        "type": "TREATS",
                        "source_dataset": "OpenTargets",
                        "validated": item.get("validated", False),
                        "evidence_source": item.get("evidence_source"),
                    }
                )
                edge_keys.add(disease_edge_key)

    dump_json(GRAPH_FILE, graph)
    if sync:
        _sync_to_neo4j(graph)
    return str(GRAPH_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Neo4j-style graph artifact.")
    parser.add_argument("--sync", action="store_true", help="Attempt to sync to Neo4j if the driver is available.")
    args = parser.parse_args()
    print(run(sync=args.sync))


if __name__ == "__main__":
    main()
