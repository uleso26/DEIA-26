# Imports.
from __future__ import annotations

import argparse
from urllib.parse import quote

from core.paths import relative_runtime_path
from data.ingestion.base import append_lineage_manifest, fetch_json_response, live_ingestion_enabled, utc_now_iso, validate_records, write_raw_payload
from data.ingestion.seed_data import OPENFDA_DATA


# Module constants.
LABEL_ENDPOINT = "https://api.fda.gov/drug/label.json"
EVENT_ENDPOINT = "https://api.fda.gov/drug/event.json"


# Normalize text values.
def _normalize_text_values(value: object, fallback: list[str]) -> list[str]:
    if isinstance(value, str):
        items = [part.strip(" -*\n\t") for part in value.replace("\r", "\n").split("\n") if part.strip()]
        return items[:3] or fallback
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_normalize_text_values(item, []))
        return flattened[:3] or fallback
    return fallback


# Format label version.
def _format_label_version(raw_value: str | None, fallback: str) -> str:
    if raw_value and len(raw_value) >= 6 and raw_value[:6].isdigit():
        return f"{raw_value[:4]}-{raw_value[4:6]}"
    return fallback


# Signal strength.
def _signal_strength(count: int) -> str:
    if count >= 500:
        return "high_reporting_volume"
    if count >= 100:
        return "moderate"
    if count >= 25:
        return "early_signal"
    return "low_volume_signal"


# Fetch live label.
def _fetch_live_label(seed_label: dict[str, object]) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    requests: list[dict[str, object]] = []
    search_candidates = [
        ("openfda.generic_name", str(seed_label["canonical_drug"])),
        *[("openfda.brand_name", brand) for brand in seed_label.get("brand_names", [])],
    ]
    for field_name, search_term in search_candidates:
        search_expression = quote(f'{field_name}:"{search_term}"', safe='')
        url = f"{LABEL_ENDPOINT}?search={search_expression}&limit=1"
        response = fetch_json_response(url)
        payload = response.get("payload") if response.get("ok") else None
        results = payload.get("results", []) if isinstance(payload, dict) else []
        requests.append(
            {
                "dataset": "drug_labels",
                "url": url,
                "field": field_name,
                "search_term": search_term,
                "status_code": response.get("status_code"),
                "ok": response.get("ok"),
                "record_count": len(results),
                "error": response.get("error"),
            }
        )
        if not results:
            continue
        record = results[0]
        openfda = record.get("openfda", {})
        return (
            {
                "canonical_drug": seed_label["canonical_drug"],
                "brand_names": openfda.get("brand_name", seed_label.get("brand_names", []))[:4],
                "label_version": _format_label_version(record.get("effective_time"), str(seed_label["label_version"])),
                "indications": _normalize_text_values(
                    record.get("indications_and_usage"),
                    list(seed_label.get("indications", [])),
                ),
                "warnings": _normalize_text_values(
                    record.get("warnings")
                    or record.get("warnings_and_cautions")
                    or record.get("boxed_warning"),
                    list(seed_label.get("warnings", [])),
                ),
                "source_url": LABEL_ENDPOINT,
            },
            requests,
        )
    return None, requests


# Fetch live events.
def _fetch_live_events(seed_label: dict[str, object]) -> tuple[list[dict[str, object]] | None, list[dict[str, object]]]:
    requests: list[dict[str, object]] = []
    search_terms = [str(seed_label["canonical_drug"]), *[str(brand) for brand in seed_label.get("brand_names", [])]]
    for search_term in search_terms:
        search_expression = quote(f'patient.drug.medicinalproduct:"{search_term.upper()}"', safe='')
        url = f"{EVENT_ENDPOINT}?search={search_expression}&count=patient.reaction.reactionmeddrapt.exact&limit=5"
        response = fetch_json_response(url)
        payload = response.get("payload") if response.get("ok") else None
        results = payload.get("results", []) if isinstance(payload, dict) else []
        requests.append(
            {
                "dataset": "adverse_events",
                "url": url,
                "search_term": search_term,
                "status_code": response.get("status_code"),
                "ok": response.get("ok"),
                "record_count": len(results),
                "error": response.get("error"),
            }
        )
        if not results:
            continue
        time_period = f"captured_{utc_now_iso()[:10]}"
        return (
            [
                {
                    "canonical_drug": seed_label["canonical_drug"],
                    "event": str(item.get("term", "")).lower(),
                    "count": int(item.get("count", 0)),
                    "subgroup": "all_reports",
                    "time_period": time_period,
                    "signal_strength": _signal_strength(int(item.get("count", 0))),
                    "source": "OpenFDA FAERS",
                }
                for item in results
                if item.get("term") and item.get("count") is not None
            ],
            requests,
        )
    return None, requests


# Run.
def run() -> dict[str, str]:
    use_live = live_ingestion_enabled("openfda")
    labels: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    request_log: list[dict[str, object]] = []

    if use_live:
        for seed_label in OPENFDA_DATA["drug_labels"]:
            live_label, label_requests = _fetch_live_label(seed_label)
            live_events, event_requests = _fetch_live_events(seed_label)
            request_log.extend(label_requests)
            request_log.extend(event_requests)
            labels.append(live_label or dict(seed_label))
            if live_events:
                events.extend(live_events)
            else:
                events.extend(
                    [
                        dict(item)
                        for item in OPENFDA_DATA["adverse_events"]
                        if item["canonical_drug"] == seed_label["canonical_drug"]
                    ]
                )
    else:
        labels = [dict(item) for item in OPENFDA_DATA["drug_labels"]]
        events = [dict(item) for item in OPENFDA_DATA["adverse_events"]]

    mode = "live_api" if use_live and any(entry.get("ok") and entry.get("record_count") for entry in request_log) else "seed_fallback"
    if mode == "seed_fallback":
        labels = [dict(item) for item in OPENFDA_DATA["drug_labels"]]
        events = [dict(item) for item in OPENFDA_DATA["adverse_events"]]

    labels = validate_records(
        labels,
        ["canonical_drug", "brand_names", "label_version", "indications", "warnings", "source_url"],
        "openfda.drug_labels",
    )
    events = validate_records(
        events,
        ["canonical_drug", "event", "count", "subgroup", "time_period", "signal_strength", "source"],
        "openfda.adverse_events",
    )

    events_path = write_raw_payload("openfda_adverse_events.json", events)
    labels_path = write_raw_payload("drug_labels.json", labels)
    append_lineage_manifest(
        "openfda",
        {
            "mode": mode,
            "raw_files": {"adverse_events": relative_runtime_path(events_path), "drug_labels": relative_runtime_path(labels_path)},
            "record_counts": {"adverse_events": len(events), "drug_labels": len(labels)},
            "upstream_requests": request_log,
        },
    )
    return {"adverse_events": str(events_path), "drug_labels": str(labels_path)}


# Main.
def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed OpenFDA payloads.")
    parser.parse_args()
    result = run()
    print(result)


# CLI entrypoint.
if __name__ == "__main__":
    main()
