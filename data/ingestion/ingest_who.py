# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import argparse
import os
from urllib.parse import quote

from core.paths import relative_runtime_path
from data.ingestion.base import append_lineage_manifest, fetch_json_response, live_ingestion_enabled, utc_now_iso, validate_records, write_raw_payload
from data.ingestion.seed_data import WHO_GHO_DATA


# Define the constants lookup tables and settings used below
COUNTRY_CODES = {
    "United Kingdom": "GBR",
    "United States": "USA",
}


# Extract values from the upstream payload
def _extract_values(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("value"), list):
            return [item for item in payload["value"] if isinstance(item, dict)]
        if isinstance(payload.get("d", {}).get("results"), list):
            return [item for item in payload["d"]["results"] if isinstance(item, dict)]
    return []


# Look up the WHO indicator configuration for a requested measure
def _lookup_indicator() -> tuple[dict[str, object] | None, dict[str, object]]:
    base_url = os.getenv("WHO_GHO_BASE_URL", "https://ghoapi.azureedge.net/api").rstrip("/")
    query = quote("contains(IndicatorName,'Diabetes prevalence')", safe="(),'")
    url = f"{base_url}/Indicator?$filter={query}&$format=json"
    response = fetch_json_response(url)
    candidates = _extract_values(response.get("payload"))
    fallback_response = None
    if not candidates:
        fallback_url = f"{base_url}/Indicator?$format=json"
        fallback_response = fetch_json_response(fallback_url)
        all_candidates = _extract_values(fallback_response.get("payload"))
        candidates = [
            item
            for item in all_candidates
            if "diabetes" in str(item.get("IndicatorName", "")).lower()
            and "prevalence" in str(item.get("IndicatorName", "")).lower()
        ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            "age-standardized" in str(item.get("IndicatorName", "")).lower(),
            "adults aged 18 years and older" in str(item.get("IndicatorName", "")).lower(),
            str(item.get("IndicatorName", "")).lower().count("diabetes"),
        ),
        reverse=True,
    )
    chosen = ranked[0] if ranked else None
    return (
        chosen,
        {
            "dataset": "who_indicator_lookup",
            "url": url,
            "status_code": response.get("status_code"),
            "ok": response.get("ok"),
            "record_count": len(candidates),
            "error": response.get("error"),
            "fallback_status_code": fallback_response.get("status_code") if fallback_response else None,
            "fallback_ok": fallback_response.get("ok") if fallback_response else None,
        },
    )


# Fetch country record from the configured source
def _fetch_country_record(seed_record: dict[str, object], indicator_code: str) -> tuple[dict[str, object] | None, dict[str, object]]:
    country = str(seed_record["country"])
    country_code = COUNTRY_CODES.get(country)
    base_url = os.getenv("WHO_GHO_BASE_URL", "https://ghoapi.azureedge.net/api").rstrip("/")
    if not country_code:
        return None, {
            "dataset": "who_country",
            "url": None,
            "country": country,
            "status_code": None,
            "ok": False,
            "record_count": 0,
            "error": "No country code mapping configured",
        }

    filter_query = quote(f"SpatialDim eq '{country_code}' and Dim1 eq 'SEX_BTSX'", safe="='")
    order_query = quote("TimeDim desc")
    url = f"{base_url}/{indicator_code}?$filter={filter_query}&$orderby={order_query}&$top=1&$format=json"
    response = fetch_json_response(url)
    values = _extract_values(response.get("payload"))
    record = values[0] if values else None
    request_log = {
        "dataset": "who_country",
        "url": url,
        "country": country,
        "country_code": country_code,
        "status_code": response.get("status_code"),
        "ok": response.get("ok"),
        "record_count": len(values),
        "error": response.get("error"),
    }
    if not isinstance(record, dict):
        return None, request_log

    numeric_value = record.get("NumericValue")
    if numeric_value is None:
        try:
            numeric_value = float(record.get("Value"))
        except (TypeError, ValueError):
            numeric_value = None
    if numeric_value is None:
        return None, request_log

    return (
        {
            "country": country,
            "country_code": country_code,
            "indicator": str(record.get("IndicatorName") or seed_record["indicator"]),
            "indicator_code": indicator_code,
            "year": int(record.get("TimeDim") or seed_record["year"]),
            "value": float(numeric_value),
            "unit": str(seed_record["unit"]),
            "source_url": url,
            "ingested_at": utc_now_iso(),
            "data_mode": "live_api",
        },
        request_log,
    )


# Run the main workflow implemented by this module
def run() -> str:
    use_live = live_ingestion_enabled("who")
    request_log: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    accepted_live_records = 0

    indicator = None
    if use_live:
        indicator, indicator_request = _lookup_indicator()
        request_log.append(indicator_request)

    indicator_code = str(indicator.get("IndicatorCode")) if isinstance(indicator, dict) and indicator.get("IndicatorCode") else None
    for seed_record in WHO_GHO_DATA:
        if use_live and indicator_code:
            live_record, request = _fetch_country_record(seed_record, indicator_code)
            request_log.append(request)
            if live_record:
                accepted_live_records += 1
                records.append(live_record)
                continue
        fallback = dict(seed_record)
        fallback["ingested_at"] = utc_now_iso()
        fallback["data_mode"] = "seed_fixture"
        records.append(fallback)

    mode = "live_api" if use_live and accepted_live_records else "seed_fixture"
    records = validate_records(
        records,
        ["country", "indicator", "year", "value", "unit"],
        "who",
    )
    path = write_raw_payload("who_gho.json", records)
    append_lineage_manifest(
        "who",
        {
            "mode": mode,
            "raw_files": {"who_gho": relative_runtime_path(path)},
            "record_counts": {"who_gho": len(records)},
            "accepted_live_records": accepted_live_records,
            "upstream_requests": request_log,
        },
    )
    return str(path)


# Coordinate the main execution path for this module
def main() -> None:
    parser = argparse.ArgumentParser(description="Write WHO GHO payload with optional live refresh.")
    parser.parse_args()
    print(run())


# CLI entrypoint
if __name__ == "__main__":
    main()
