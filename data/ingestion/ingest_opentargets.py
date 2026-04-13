# Imports.
from __future__ import annotations

import argparse
import os

from core.paths import relative_runtime_path
from data.ingestion.base import (
    append_lineage_manifest,
    fetch_json_response,
    live_ingestion_enabled,
    post_json_response,
    utc_now_iso,
    validate_records,
    write_raw_payload,
)
from data.ingestion.seed_data import OPENTARGETS_DATA


# Module constants.
TARGET_QUERY = """
query TargetSummary($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
  }
}
"""


# Lookup ensembl id.
def _lookup_ensembl_id(symbol: str) -> tuple[str | None, dict[str, object]]:
    base_url = os.getenv("ENSEMBL_LOOKUP_BASE_URL", "https://rest.ensembl.org/lookup/symbol/homo_sapiens").rstrip("/")
    url = f"{base_url}/{symbol}?content-type=application/json"
    response = fetch_json_response(url, headers={"Content-Type": "application/json"})
    payload = response.get("payload") if response.get("ok") else None
    ensembl_id = payload.get("id") if isinstance(payload, dict) else None
    return (
        str(ensembl_id) if ensembl_id else None,
        {
            "dataset": "ensembl_lookup",
            "url": url,
            "target_symbol": symbol,
            "status_code": response.get("status_code"),
            "ok": response.get("ok"),
            "record_count": 1 if ensembl_id else 0,
            "error": response.get("error"),
        },
    )


# Fetch live target.
def _fetch_live_target(symbol: str) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    ensembl_id, lookup_request = _lookup_ensembl_id(symbol)
    request_log = [lookup_request]
    if not ensembl_id:
        return None, request_log

    graphql_url = os.getenv("OPENTARGETS_GRAPHQL_URL", "https://api.platform.opentargets.org/api/v4/graphql").rstrip("/")
    response = post_json_response(
        graphql_url,
        {"query": TARGET_QUERY, "variables": {"ensemblId": ensembl_id}},
    )
    payload = response.get("payload") if response.get("ok") else None
    target = payload.get("data", {}).get("target") if isinstance(payload, dict) else None
    request_log.append(
        {
            "dataset": "opentargets_target",
            "url": graphql_url,
            "target_symbol": symbol,
            "ensembl_id": ensembl_id,
            "status_code": response.get("status_code"),
            "ok": response.get("ok"),
            "record_count": 1 if target else 0,
            "error": response.get("error"),
        }
    )
    if not isinstance(target, dict):
        return None, request_log

    return (
        {
            "ensembl_id": str(target.get("id") or ensembl_id),
            "approved_symbol": str(target.get("approvedSymbol") or symbol),
            "approved_name": str(target.get("approvedName") or symbol),
        },
        request_log,
    )


# Run.
def run() -> str:
    use_live = live_ingestion_enabled("opentargets")
    timestamp = utc_now_iso()
    request_log: list[dict[str, object]] = []
    live_targets: dict[str, dict[str, object]] = {}

    if use_live:
        for symbol in sorted({str(item["canonical_target"]) for item in OPENTARGETS_DATA}):
            live_target, requests = _fetch_live_target(symbol)
            request_log.extend(requests)
            if live_target:
                live_targets[symbol] = live_target

    records: list[dict[str, object]] = []
    accepted_live_records = 0
    for record in OPENTARGETS_DATA:
        live_target = live_targets.get(str(record["canonical_target"]))
        normalized = dict(record)
        normalized["ingested_at"] = timestamp
        if live_target:
            accepted_live_records += 1
            # Open Targets gives us live target identity. We keep the curated
            # drug-target mapping so downstream questions stay stable.
            normalized.update(
                {
                    "canonical_target": live_target["approved_symbol"],
                    "target_name": live_target["approved_name"],
                    "ensembl_id": live_target["ensembl_id"],
                    "source_url": f"https://platform.opentargets.org/target/{live_target['ensembl_id']}",
                    "evidence_source": "Open Targets GraphQL target metadata + curated drug-target map",
                    "data_mode": "live_target_metadata_plus_seed_mapping",
                }
            )
        else:
            normalized["data_mode"] = "seed_fixture"
        records.append(normalized)

    mode = "live_api" if use_live and accepted_live_records else "seed_fixture"
    records = validate_records(
        records,
        ["canonical_drug", "canonical_target", "association_type", "evidence_source"],
        "opentargets",
    )
    path = write_raw_payload("opentargets.json", records)
    append_lineage_manifest(
        "opentargets",
        {
            "mode": mode,
            "raw_files": {"opentargets": relative_runtime_path(path)},
            "record_counts": {"opentargets": len(records)},
            "accepted_live_records": accepted_live_records,
            "upstream_requests": request_log,
        },
    )
    return str(path)


# Main.
def main() -> None:
    parser = argparse.ArgumentParser(description="Write OpenTargets payload with optional live target refresh.")
    parser.parse_args()
    print(run())


# CLI entrypoint.
if __name__ == "__main__":
    main()
