from __future__ import annotations

import argparse
import os
import re

from core.paths import relative_runtime_path
from data.ingestion.base import (
    append_lineage_manifest,
    fetch_json_response,
    fetch_text_response,
    live_ingestion_enabled,
    utc_now_iso,
    write_raw_payload,
)
from data.ingestion.seed_data import DRUGBANK_OPEN_DATA


def _first_match(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
    return None


def _extract_drug_class(html: str, fallback: str) -> str:
    for label in [
        "Dual GIP/GLP-1 receptor agonist",
        "Dual GIP and GLP-1 receptor agonist",
        "GLP-1 receptor agonist",
        "GLP-1 Agonists",
        "SGLT2 inhibitor",
        "Sodium-glucose cotransporter 2 inhibitor",
    ]:
        if label.lower() in html.lower():
            return label
    return fallback


def _extract_api_drug_class(payload: dict[str, object], fallback: str) -> str:
    for key in ["indication", "description", "mechanism_of_action"]:
        value = payload.get(key)
        if isinstance(value, str):
            lowered = value.lower()
            if "glp-1" in lowered and "gip" in lowered:
                return "Dual GIP/GLP-1 receptor agonist"
            if "glp-1" in lowered:
                return "GLP-1 receptor agonist"
            if "sglt2" in lowered or "sodium-glucose cotransporter 2" in lowered:
                return "SGLT2 inhibitor"
    return fallback


def _fetch_live_api_drug(seed_record: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    api_key = os.getenv("DRUGBANK_API_KEY", "").strip()
    if not api_key:
        return None, None

    base_url = os.getenv("DRUGBANK_API_BASE_URL", "https://api.drugbank.com/v1/drugs").rstrip("/")
    api_url = f"{base_url}/{seed_record['drugbank_id']}"
    response = fetch_json_response(
        api_url,
        headers={
            "Authorization": api_key,
            "Accept": "application/json",
            "User-Agent": "T2DIntelligencePlatform/1.0",
        },
    )
    payload = response.get("payload") if response.get("ok") else None
    request_log = {
        "dataset": "drugbank_api",
        "url": api_url,
        "drugbank_id": seed_record["drugbank_id"],
        "status_code": response.get("status_code"),
        "ok": response.get("ok"),
        "record_count": 1 if isinstance(payload, dict) and payload else 0,
        "error": response.get("error"),
    }
    if not isinstance(payload, dict) or not payload:
        return None, request_log

    normalized = {
        "canonical_drug": seed_record["canonical_drug"],
        "drugbank_id": seed_record["drugbank_id"],
        "drug_class": _extract_api_drug_class(payload, str(seed_record["drug_class"])),
        "pharmacology": payload.get("description") or payload.get("indication") or seed_record["pharmacology"],
        "interactions": list(seed_record.get("interactions", [])),
        "source_url": f"https://go.drugbank.com/drugs/{seed_record['drugbank_id']}",
        "page_title": payload.get("name") or str(seed_record["canonical_drug"]).title(),
        "ingested_at": utc_now_iso(),
        "data_mode": "live_api_plus_seed_mapping",
    }
    return normalized, request_log


def _fetch_live_page_drug(seed_record: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object]]:
    page_url = f"https://go.drugbank.com/drugs/{seed_record['drugbank_id']}"
    response = fetch_text_response(
        page_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://go.drugbank.com/",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    html = response.get("payload") if response.get("ok") else None
    request_log = {
        "dataset": "drugbank_open_page",
        "url": page_url,
        "drugbank_id": seed_record["drugbank_id"],
        "status_code": response.get("status_code"),
        "ok": response.get("ok"),
        "record_count": 1 if isinstance(html, str) and html else 0,
        "error": response.get("error"),
        "blocked_reason": "cloudflare_challenge" if response.get("status_code") == 403 else None,
    }
    if not isinstance(html, str) or not html:
        return None, request_log

    title = _first_match([r"<title>(.*?)</title>", r"<h1[^>]*>(.*?)</h1>"], html)
    description = _first_match(
        [
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
        ],
        html,
    )
    normalized = {
        "canonical_drug": seed_record["canonical_drug"],
        "drugbank_id": seed_record["drugbank_id"],
        "drug_class": _extract_drug_class(html, str(seed_record["drug_class"])),
        "pharmacology": description or seed_record["pharmacology"],
        "interactions": list(seed_record.get("interactions", [])),
        "source_url": page_url,
        "page_title": title or str(seed_record["canonical_drug"]).title(),
        "ingested_at": utc_now_iso(),
        "data_mode": "live_page_plus_seed_mapping",
    }
    return normalized, request_log


def run() -> str:
    use_live = live_ingestion_enabled("drugbank_open")
    request_log: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    accepted_live_records = 0
    live_modes: set[str] = set()

    for seed_record in DRUGBANK_OPEN_DATA:
        if use_live:
            live_record, request = _fetch_live_api_drug(seed_record)
            if request:
                request_log.append(request)
            if live_record:
                accepted_live_records += 1
                live_modes.add("live_api")
                records.append(live_record)
                continue

            live_record, request = _fetch_live_page_drug(seed_record)
            request_log.append(request)
            if live_record:
                accepted_live_records += 1
                live_modes.add("live_scrape")
                records.append(live_record)
                continue
        fallback = dict(seed_record)
        fallback["ingested_at"] = utc_now_iso()
        fallback["data_mode"] = "seed_fixture"
        records.append(fallback)

    blocked_requests = [request for request in request_log if request.get("status_code") == 403]
    if live_modes:
        mode = "+".join(sorted(live_modes))
    elif use_live and blocked_requests:
        mode = "blocked_live_source_fallback"
    else:
        mode = "seed_fixture"
    path = write_raw_payload("drugbank_open.json", records)
    append_lineage_manifest(
        "drugbank_open",
        {
            "mode": mode,
            "raw_files": {"drugbank_open": relative_runtime_path(path)},
            "record_counts": {"drugbank_open": len(records)},
            "accepted_live_records": accepted_live_records,
            "blocked_live_requests": len(blocked_requests),
            "upstream_requests": request_log,
        },
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write DrugBank Open payload with optional live page scrape.")
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
