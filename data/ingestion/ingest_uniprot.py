from __future__ import annotations

import argparse
import os

from core.paths import relative_runtime_path
from data.ingestion.base import append_lineage_manifest, fetch_json_response, live_ingestion_enabled, utc_now_iso, write_raw_payload
from data.ingestion.seed_data import UNIPROT_DATA


def _extract_function(comments: list[dict[str, object]], fallback: str) -> str:
    for comment in comments:
        comment_type = str(comment.get("commentType") or comment.get("type") or "")
        if comment_type.upper() != "FUNCTION":
            continue
        texts = comment.get("texts", [])
        if isinstance(texts, list) and texts:
            value = texts[0].get("value")
            if value:
                return str(value)
    return fallback


def _fetch_live_target(seed_record: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object]]:
    base_url = os.getenv("UNIPROT_BASE_URL", "https://rest.uniprot.org/uniprotkb").rstrip("/")
    accession = str(seed_record["uniprot_id"])
    url = f"{base_url}/{accession}.json"
    response = fetch_json_response(url)
    payload = response.get("payload") if response.get("ok") else None
    protein_description = payload.get("proteinDescription", {}) if isinstance(payload, dict) else {}
    recommended = protein_description.get("recommendedName", {}) if isinstance(protein_description, dict) else {}
    full_name = recommended.get("fullName", {}) if isinstance(recommended, dict) else {}
    name_value = full_name.get("value") if isinstance(full_name, dict) else None
    has_record = bool(payload and accession == payload.get("primaryAccession"))
    request_log = {
        "dataset": "uniprot",
        "url": url,
        "uniprot_id": accession,
        "status_code": response.get("status_code"),
        "ok": response.get("ok"),
        "record_count": 1 if has_record else 0,
        "error": response.get("error"),
    }
    if not has_record:
        return None, request_log
    # Keep the target mapping from the curated layer, but refresh the protein
    # name and function text from UniProt when the live entry is available.
    normalized = {
        "canonical_target": seed_record["canonical_target"],
        "uniprot_id": accession,
        "protein_name": str(name_value or seed_record["protein_name"]),
        "function": _extract_function(list(payload.get("comments", [])), str(seed_record["function"])),
        "ingested_at": utc_now_iso(),
        "data_mode": "live_metadata_plus_seed_mapping",
    }
    return normalized, request_log


def run() -> str:
    use_live = live_ingestion_enabled("uniprot")
    records: list[dict[str, object]] = []
    request_log: list[dict[str, object]] = []
    accepted_live_records = 0

    if use_live:
        for seed_record in UNIPROT_DATA:
            live_record, request = _fetch_live_target(seed_record)
            request_log.append(request)
            if live_record:
                accepted_live_records += 1
            records.append(live_record or {**seed_record, "ingested_at": utc_now_iso(), "data_mode": "seed_fixture"})
    else:
        records = [{**record, "ingested_at": utc_now_iso(), "data_mode": "seed_fixture"} for record in UNIPROT_DATA]

    mode = "live_api" if use_live and accepted_live_records else "seed_fixture"
    path = write_raw_payload("uniprot.json", records)
    append_lineage_manifest(
        "uniprot",
        {
            "mode": mode,
            "raw_files": {"uniprot": relative_runtime_path(path)},
            "record_counts": {"uniprot": len(records)},
            "accepted_live_records": accepted_live_records,
            "upstream_requests": request_log,
        },
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed UniProt payload.")
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
