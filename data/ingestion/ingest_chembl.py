from __future__ import annotations

import argparse
import os

from core.paths import relative_runtime_path
from data.ingestion.base import append_lineage_manifest, fetch_json_response, live_ingestion_enabled, utc_now_iso, write_raw_payload
from data.ingestion.seed_data import CHEMBL_DATA


def _infer_status(max_phase: int, fallback: str) -> str:
    if max_phase >= 4:
        return "marketed"
    if max_phase == 3:
        return "late-stage development"
    return fallback


def _normalize_max_phase(value: object, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _fetch_live_chembl(seed_record: dict[str, object]) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    base_url = os.getenv("CHEMBL_BASE_URL", "https://www.ebi.ac.uk/chembl/api/data").rstrip("/")
    chembl_id = str(seed_record["chembl_id"])
    molecule_url = f"{base_url}/molecule/{chembl_id}.json"
    mechanism_url = f"{base_url}/mechanism.json?molecule_chembl_id={chembl_id}&format=json"

    molecule_response = fetch_json_response(molecule_url)
    mechanism_response = fetch_json_response(mechanism_url)
    molecule_payload = molecule_response.get("payload") if molecule_response.get("ok") else None
    mechanism_payload = mechanism_response.get("payload") if mechanism_response.get("ok") else None
    mechanisms = mechanism_payload.get("mechanisms", []) if isinstance(mechanism_payload, dict) else []
    request_log = [
        {
            "dataset": "chembl_molecule",
            "url": molecule_url,
            "chembl_id": chembl_id,
            "status_code": molecule_response.get("status_code"),
            "ok": molecule_response.get("ok"),
            "record_count": 1 if molecule_payload else 0,
            "error": molecule_response.get("error"),
        },
        {
            "dataset": "chembl_mechanism",
            "url": mechanism_url,
            "chembl_id": chembl_id,
            "status_code": mechanism_response.get("status_code"),
            "ok": mechanism_response.get("ok"),
            "record_count": len(mechanisms),
            "error": mechanism_response.get("error"),
        },
    ]
    if not isinstance(molecule_payload, dict):
        return None, request_log

    # ChEMBL gives us clean live phase metadata. We still keep the curated drug
    # mapping so downstream agents stay on canonical IDs.
    mechanism = str(
        (mechanisms[0].get("mechanism_of_action") if mechanisms else None)
        or (mechanisms[0].get("action_type") if mechanisms else None)
        or seed_record["mechanism"]
    )
    max_phase = _normalize_max_phase(molecule_payload.get("max_phase"), int(seed_record["max_phase"]))
    normalized = {
        "canonical_drug": seed_record["canonical_drug"],
        "chembl_id": str(molecule_payload.get("molecule_chembl_id") or chembl_id),
        "mechanism": mechanism,
        "max_phase": max_phase,
        "development_status": _infer_status(max_phase, str(seed_record["development_status"])),
        "ingested_at": utc_now_iso(),
        "data_mode": "live_metadata_plus_mechanism",
    }
    return normalized, request_log


def run() -> str:
    use_live = live_ingestion_enabled("chembl")
    records: list[dict[str, object]] = []
    request_log: list[dict[str, object]] = []
    accepted_live_records = 0

    if use_live:
        for seed_record in CHEMBL_DATA:
            live_record, requests = _fetch_live_chembl(seed_record)
            request_log.extend(requests)
            if live_record:
                accepted_live_records += 1
            records.append(live_record or {**seed_record, "ingested_at": utc_now_iso(), "data_mode": "seed_fixture"})
    else:
        records = [{**record, "ingested_at": utc_now_iso(), "data_mode": "seed_fixture"} for record in CHEMBL_DATA]

    mode = "live_api" if use_live and accepted_live_records else "seed_fixture"
    path = write_raw_payload("chembl.json", records)
    append_lineage_manifest(
        "chembl",
        {
            "mode": mode,
            "raw_files": {"chembl": relative_runtime_path(path)},
            "record_counts": {"chembl": len(records)},
            "accepted_live_records": accepted_live_records,
            "upstream_requests": request_log,
        },
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed ChEMBL payload.")
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
