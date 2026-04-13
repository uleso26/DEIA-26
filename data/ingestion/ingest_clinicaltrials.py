# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import argparse
import os

from core.paths import relative_runtime_path
from data.ingestion.base import append_lineage_manifest, fetch_json_response, live_ingestion_enabled, utc_now_iso, validate_records, write_raw_payload
from data.ingestion.seed_data import CLINICAL_TRIALS


# Normalize phase before reuse in the pipeline
def _normalize_phase(values: list[str], fallback: str) -> str:
    if not values:
        return fallback
    raw_value = str(values[0]).upper().replace("_", "")
    if raw_value.startswith("PHASE"):
        suffix = raw_value.removeprefix("PHASE")
        return f"Phase {suffix}" if suffix else fallback
    return str(values[0]).replace("_", " ").title()


# Normalize status before reuse in the pipeline
def _normalize_status(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return value.replace("_", " ").title()


# Extract interventions from the upstream payload
def _extract_interventions(items: list[dict[str, object]], fallback: list[str]) -> list[str]:
    names = [str(item.get("name")) for item in items if item.get("name")]
    return names or fallback


# Extract primary endpoint from the upstream payload
def _extract_primary_endpoint(items: list[dict[str, object]], fallback: str) -> str:
    if not items:
        return fallback
    measure = items[0].get("measure")
    time_frame = items[0].get("timeFrame")
    if measure and time_frame:
        return f"{measure} ({time_frame})"
    if measure:
        return str(measure)
    return fallback


# Extract publication pmid from the upstream payload
def _extract_publication_pmid(items: list[dict[str, object]], trial_name: str, fallback: str) -> str:
    for item in items:
        pmid = item.get("pmid")
        if pmid and str(item.get("type", "")).upper() == "RESULT":
            return f"PMID{pmid}" if not str(pmid).startswith("PMID") else str(pmid)

    trial_marker = trial_name.lower()
    exclusion_terms = ["post-hoc", "substudy", "mri", "cgm", "indirect", "cost", "threshold"]
    best_pmid = ""
    best_score = -999
    for item in items:
        pmid = item.get("pmid")
        citation = str(item.get("citation", "")).lower()
        if not pmid:
            continue
        score = 0
        if trial_marker and trial_marker in citation:
            score += 5
        if "versus" in citation:
            score += 3
        if "phase 3" in citation or "trial" in citation:
            score += 1
        if any(term in citation for term in exclusion_terms):
            score -= 4
        if score > best_score:
            best_score = score
            best_pmid = str(pmid)

    if best_pmid:
        return f"PMID{best_pmid}" if not best_pmid.startswith("PMID") else best_pmid

    for item in items:
        pmid = item.get("pmid")
        if pmid:
            return f"PMID{pmid}" if not str(pmid).startswith("PMID") else str(pmid)
    return fallback


# Fetch live trial from the configured source
def _fetch_live_trial(seed_trial: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object]]:
    base_url = os.getenv("CLINICALTRIALS_BASE_URL", "https://clinicaltrials.gov/api/v2").rstrip("/")
    url = f"{base_url}/studies/{seed_trial['nct_id']}"
    response = fetch_json_response(url)
    payload = response.get("payload") if response.get("ok") else None
    protocol = payload.get("protocolSection", {}) if isinstance(payload, dict) else {}
    identification = protocol.get("identificationModule", {}) if isinstance(protocol, dict) else {}
    status = protocol.get("statusModule", {}) if isinstance(protocol, dict) else {}
    conditions = protocol.get("conditionsModule", {}) if isinstance(protocol, dict) else {}
    arms = protocol.get("armsInterventionsModule", {}) if isinstance(protocol, dict) else {}
    outcomes = protocol.get("outcomesModule", {}) if isinstance(protocol, dict) else {}
    references = protocol.get("referencesModule", {}) if isinstance(protocol, dict) else {}
    design = protocol.get("designModule", {}) if isinstance(protocol, dict) else {}
    has_record = bool(identification.get("nctId"))
    request_log = {
        "dataset": "clinical_trials",
        "url": url,
        "nct_id": seed_trial["nct_id"],
        "status_code": response.get("status_code"),
        "ok": response.get("ok"),
        "record_count": 1 if has_record else 0,
        "error": response.get("error"),
    }
    if not has_record:
        return None, request_log
    normalized = {
        "nct_id": str(identification.get("nctId", seed_trial["nct_id"])),
        "trial_name": str(identification.get("acronym") or seed_trial["trial_name"]),
        "official_title": str(
            identification.get("officialTitle")
            or identification.get("briefTitle")
            or seed_trial["trial_name"]
        ),
        "phase": _normalize_phase(design.get("phases", []), str(seed_trial["phase"])),
        "status": _normalize_status(status.get("overallStatus"), str(seed_trial["status"])),
        "conditions": list(conditions.get("conditions", list(seed_trial.get("conditions", [])))),
        "interventions": _extract_interventions(
            list(arms.get("interventions", [])),
            list(seed_trial.get("interventions", [])),
        ),
        "comparison_type": seed_trial["comparison_type"],
        "primary_endpoint": _extract_primary_endpoint(
            list(outcomes.get("primaryOutcomes", [])),
            str(seed_trial["primary_endpoint"]),
        ),
        "results": dict(seed_trial["results"]),
        "publication_reference": _extract_publication_pmid(
            list(references.get("references", [])),
            str(identification.get("acronym") or seed_trial["trial_name"]),
            str(seed_trial.get("publication_reference", "")),
        ),
        "source_url": f"https://clinicaltrials.gov/study/{identification.get('nctId', seed_trial['nct_id'])}",
        "completion_date": str(
            status.get("completionDateStruct", {}).get("date") or seed_trial["completion_date"]
        ),
        "api_version": "ClinicalTrials.gov v2 live study endpoint + curated efficacy results",
        "ingested_at": utc_now_iso(),
        "data_mode": "live_metadata_plus_curated_results",
    }
    return normalized, request_log


# Build the seeded trial record used in offline mode
def _seed_trial(seed_trial: dict[str, object]) -> dict[str, object]:
    normalized = dict(seed_trial)
    normalized["ingested_at"] = utc_now_iso()
    normalized["data_mode"] = "seed_curated_results"
    return normalized


# Run the main workflow implemented by this module
def run() -> str:
    use_live = live_ingestion_enabled("clinicaltrials")
    normalized_trials: list[dict[str, object]] = []
    request_log: list[dict[str, object]] = []

    if use_live:
        for seed_trial in CLINICAL_TRIALS:
            live_trial, request = _fetch_live_trial(seed_trial)
            request_log.append(request)
            normalized_trials.append(live_trial or _seed_trial(seed_trial))
    else:
        normalized_trials = [_seed_trial(item) for item in CLINICAL_TRIALS]

    mode = "live_api" if use_live and any(entry.get("ok") and entry.get("record_count") for entry in request_log) else "seed_fallback"
    if mode == "seed_fallback":
        normalized_trials = [_seed_trial(item) for item in CLINICAL_TRIALS]

    normalized_trials = validate_records(
        normalized_trials,
        [
            "nct_id",
            "trial_name",
            "phase",
            "status",
            "conditions",
            "interventions",
            "primary_endpoint",
            "results",
            "source_url",
        ],
        "clinicaltrials",
    )

    path = write_raw_payload("clinical_trials.json", normalized_trials)
    append_lineage_manifest(
        "clinicaltrials",
        {
            "mode": mode,
            "raw_files": {"clinical_trials": relative_runtime_path(path)},
            "record_counts": {"clinical_trials": len(normalized_trials)},
            "upstream_requests": request_log,
        },
    )
    return str(path)


# Coordinate the main execution path for this module
def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed ClinicalTrials.gov payload.")
    parser.parse_args()
    print(run())


# CLI entrypoint
if __name__ == "__main__":
    main()
