# Imports.
from __future__ import annotations

from typing import Any

from data.canonical.resolver import get_resolver
from mcp_servers.base_server import BaseMCPStyleServer
from core.storage import connect_sqlite


# Safety Server.
class SafetyServer(BaseMCPStyleServer):
    def __init__(self) -> None:
        super().__init__("safety-mcp-server")
        self.resolver = get_resolver()
        self.register_tool(
            "search_adverse_events",
            "Search structured FAERS-style adverse event counts for a drug.",
            self.search_adverse_events,
        )
        self.register_tool(
            "get_drug_label",
            "Return the drug label summary for a canonical drug.",
            self.get_drug_label,
        )
        self.register_tool(
            "get_safety_summary",
            "Return an aggregated safety summary for a drug and optional subgroup.",
            self.get_safety_summary,
        )

    def _canonical_drug(self, drug: str) -> dict[str, Any]:
        return self.resolver.resolve_drug(drug) or {
            "canonical_id": drug.lower(),
            "matched_aliases": [drug.lower()],
            "confidence": 0.5,
            "evidence_tier": "Tier 2",
        }

    def search_adverse_events(self, drug: str, subgroup: str | None = None, top_k: int = 5) -> dict[str, Any]:
        resolved = self._canonical_drug(drug)
        connection = connect_sqlite()
        try:
            cursor = connection.cursor()
            rows = []
            if subgroup:
                cursor.execute(
                    """
                    SELECT event, count, subgroup, time_period, signal_strength, source
                    FROM adverse_events
                    WHERE canonical_drug = ? AND subgroup = ?
                    ORDER BY count DESC
                    LIMIT ?
                    """,
                    (resolved["canonical_id"], subgroup, top_k),
                )
                rows = cursor.fetchall()
            if not rows:
                cursor.execute(
                    """
                    SELECT event, count, subgroup, time_period, signal_strength, source
                    FROM adverse_events
                    WHERE canonical_drug = ?
                    ORDER BY count DESC
                    LIMIT ?
                    """,
                    (resolved["canonical_id"], top_k),
                )
                rows = cursor.fetchall()
        finally:
            connection.close()

        records = [
            {
                "event": event,
                "count": count,
                "subgroup": subgroup_name,
                "time_period": time_period,
                "signal_strength": signal_strength,
                "source": source,
            }
            for event, count, subgroup_name, time_period, signal_strength, source in rows
        ]
        return {
            "tool_name": "search_adverse_events",
            "canonical_entities": {"drug": resolved},
            "source_metadata": {"storage": "sqlite", "dataset": "adverse_events"},
            "records": records,
            "raw_provenance": {"query_drug": drug, "subgroup": subgroup},
        }

    def get_drug_label(self, drug: str) -> dict[str, Any]:
        resolved = self._canonical_drug(drug)
        connection = connect_sqlite()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT canonical_drug, brand_names, label_version, indications, warnings, source_url
                FROM drug_labels
                WHERE canonical_drug = ?
                """,
                (resolved["canonical_id"],),
            )
            row = cursor.fetchone()
        finally:
            connection.close()

        records = []
        if row:
            records.append(
                {
                    "canonical_drug": row[0],
                    "brand_names": row[1].split(","),
                    "label_version": row[2],
                    "indications": row[3].split(","),
                    "warnings": row[4].split(","),
                    "source_url": row[5],
                }
            )
        return {
            "tool_name": "get_drug_label",
            "canonical_entities": {"drug": resolved},
            "source_metadata": {"storage": "sqlite", "dataset": "drug_labels"},
            "records": records,
            "raw_provenance": {"query_drug": drug},
        }

    def get_safety_summary(self, drug: str, subgroup: str | None = None) -> dict[str, Any]:
        events = self.search_adverse_events(drug, subgroup=subgroup, top_k=3)
        label = self.get_drug_label(drug)
        top_events = events["records"]
        warnings = label["records"][0]["warnings"] if label["records"] else []
        summary = {
            "canonical_drug": events["canonical_entities"]["drug"]["canonical_id"],
            "top_events": top_events,
            "warnings": warnings,
            "statement": "FAERS voluntary reporting patterns should be treated as signal review support rather than incidence or causality evidence.",
        }
        return {
            "tool_name": "get_safety_summary",
            "canonical_entities": events["canonical_entities"],
            "source_metadata": {"storage": "sqlite", "dataset": "adverse_events + drug_labels"},
            "records": [summary],
            "raw_provenance": {"query_drug": drug, "subgroup": subgroup},
        }


# Main.
def main() -> None:
    SafetyServer().run_cli()


# CLI entrypoint.
if __name__ == "__main__":
    main()
