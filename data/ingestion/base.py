from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.logging_utils import get_logger
from core.paths import LINEAGE_DIR, PROV_LINEAGE_DIR, RAW_DIR, ROOT, ensure_runtime_directories, relative_runtime_path
from core.runtime_utils import env_flag, utc_now_iso


logger = get_logger(__name__)


def live_ingestion_enabled(source_name: str) -> bool:
    source_flag = f"USE_LIVE_{source_name.upper()}_INGESTION"
    return env_flag(source_flag, env_flag("USE_LIVE_INGESTION", False))


def write_raw_payload(filename: str, payload: object) -> Path:
    ensure_runtime_directories()
    # Live and seed refreshes land in runtime/raw so normal project use does not
    # rewrite tracked fixture files under data/raw.
    path = RAW_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def clone_records(records: list[dict[str, object]], timestamp_fields: list[str] | None = None) -> list[dict[str, object]]:
    cloned_records = [dict(record) for record in records]
    if not timestamp_fields:
        return cloned_records
    timestamp = utc_now_iso()
    for record in cloned_records:
        for field_name in timestamp_fields:
            record[field_name] = timestamp
    return cloned_records


def validate_records(
    records: list[dict[str, object]],
    required_fields: list[str],
    source_name: str,
) -> list[dict[str, object]]:
    """Filter out malformed records while leaving a breadcrumb in the runtime logs."""
    valid_records: list[dict[str, object]] = []
    for index, record in enumerate(records):
        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in record or record[field_name] is None
        ]
        if missing_fields:
            logger.warning(
                "%s record %d missing required fields: %s",
                source_name,
                index,
                ", ".join(missing_fields),
            )
            continue
        valid_records.append(record)
    logger.info("%s validation accepted %d/%d records", source_name, len(valid_records), len(records))
    return valid_records


def write_seed_payload(
    source_name: str,
    filename: str,
    records: list[dict[str, object]],
    *,
    timestamp_fields: list[str] | None = None,
    extra_manifest: dict[str, object] | None = None,
    required_fields: list[str] | None = None,
) -> str:
    normalized_records = clone_records(records, timestamp_fields=timestamp_fields)
    if required_fields:
        normalized_records = validate_records(normalized_records, required_fields, source_name)
    path = write_raw_payload(filename, normalized_records)
    append_lineage_manifest(
        source_name,
        {
            "mode": "seed_fixture",
            "raw_files": {source_name: relative_runtime_path(path)},
            "record_counts": {source_name: len(normalized_records)},
            **(extra_manifest or {}),
        },
    )
    return str(path)


def _validated_remote_url(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return url


def _build_request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, object] | None = None,
    method: str | None = None,
) -> Request | None:
    safe_url = _validated_remote_url(url)
    if not safe_url:
        return None
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    return Request(safe_url, headers=headers or {}, data=data, method=method)


def try_fetch_json(url: str, headers: dict[str, str] | None = None) -> object | None:
    request = _build_request(url, headers=headers)
    if request is None:
        return None
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def fetch_json_response(url: str, headers: dict[str, str] | None = None) -> dict[str, object]:
    request = _build_request(url, headers=headers)
    if request is None:
        return {
            "ok": False,
            "url": url,
            "status_code": None,
            "payload": None,
            "error": "Unsupported URL scheme",
        }
    try:
        with urlopen(request, timeout=20) as response:
            return {
                "ok": True,
                "url": url,
                "status_code": getattr(response, "status", 200),
                "payload": json.loads(response.read().decode("utf-8")),
                "error": None,
            }
    except HTTPError as exc:
        return {"ok": False, "url": url, "status_code": exc.code, "payload": None, "error": str(exc)}
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "url": url, "status_code": None, "payload": None, "error": str(exc)}


def post_json_response(url: str, payload: dict[str, object], headers: dict[str, str] | None = None) -> dict[str, object]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = _build_request(url, headers=request_headers, payload=payload, method="POST")
    if request is None:
        return {
            "ok": False,
            "url": url,
            "status_code": None,
            "payload": None,
            "error": "Unsupported URL scheme",
        }
    try:
        with urlopen(request, timeout=25) as response:
            return {
                "ok": True,
                "url": url,
                "status_code": getattr(response, "status", 200),
                "payload": json.loads(response.read().decode("utf-8")),
                "error": None,
            }
    except HTTPError as exc:
        return {"ok": False, "url": url, "status_code": exc.code, "payload": None, "error": str(exc)}
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "url": url, "status_code": None, "payload": None, "error": str(exc)}


def fetch_text_response(url: str, headers: dict[str, str] | None = None) -> dict[str, object]:
    request = _build_request(url, headers=headers)
    if request is None:
        return {
            "ok": False,
            "url": url,
            "status_code": None,
            "payload": None,
            "error": "Unsupported URL scheme",
        }
    try:
        with urlopen(request, timeout=20) as response:
            return {
                "ok": True,
                "url": url,
                "status_code": getattr(response, "status", 200),
                "payload": response.read().decode("utf-8", errors="replace"),
                "error": None,
            }
    except HTTPError as exc:
        return {"ok": False, "url": url, "status_code": exc.code, "payload": None, "error": str(exc)}
    except (URLError, TimeoutError) as exc:
        return {"ok": False, "url": url, "status_code": None, "payload": None, "error": str(exc)}


def append_lineage_manifest(source_name: str, payload: dict[str, object]) -> Path:
    ensure_runtime_directories()
    path = LINEAGE_DIR / f"{source_name}.jsonl"
    normalized_payload = dict(payload)
    raw_files = normalized_payload.get("raw_files")
    if isinstance(raw_files, dict):
        cleaned_raw_files: dict[str, object] = {}
        for key, value in raw_files.items():
            if isinstance(value, str) and value.startswith(str(ROOT)):
                cleaned_raw_files[key] = relative_runtime_path(Path(value))
            else:
                cleaned_raw_files[key] = value
        normalized_payload["raw_files"] = cleaned_raw_files
    enriched = {"source_name": source_name, "recorded_at": utc_now_iso(), **normalized_payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(enriched, ensure_ascii=True) + "\n")
    append_prov_manifest(source_name, enriched)
    return path


def _prov_payload(source_name: str, payload: dict[str, object]) -> dict[str, object]:
    recorded_at = str(payload.get("recorded_at") or utc_now_iso())
    raw_files = payload.get("raw_files") or {}
    generated_entities = []
    if isinstance(raw_files, dict):
        for dataset_name, raw_file in raw_files.items():
            generated_entities.append(
                {
                    "id": f"entity:{source_name}:{dataset_name}",
                    "prov:type": "prov:Entity",
                    "name": dataset_name,
                    "location": raw_file,
                }
            )

    upstream_requests = payload.get("upstream_requests") or []
    used_entities = []
    if isinstance(upstream_requests, list):
        for index, request in enumerate(upstream_requests):
            if not isinstance(request, dict):
                continue
            used_entities.append(
                {
                    "id": f"source:{source_name}:{index}",
                    "prov:type": "prov:Entity",
                    "url": request.get("url"),
                    "status_code": request.get("status_code"),
                    "ok": request.get("ok"),
                }
            )

    activity = {
        "id": f"activity:{source_name}:{recorded_at}",
        "prov:type": "prov:Activity",
        "startedAtTime": recorded_at,
        "endedAtTime": recorded_at,
        "mode": payload.get("mode"),
        "record_counts": payload.get("record_counts"),
    }
    return {
        "prefix": {
            "prov": "http://www.w3.org/ns/prov#",
        },
        "activity": activity,
        "used": used_entities,
        "generated": generated_entities,
        "wasAssociatedWith": {
            "id": f"agent:ingestion:{source_name}",
            "prov:type": "prov:SoftwareAgent",
            "label": f"{source_name} ingestion pipeline",
        },
    }


def append_prov_manifest(source_name: str, payload: dict[str, object]) -> Path:
    ensure_runtime_directories()
    path = PROV_LINEAGE_DIR / f"{source_name}.jsonl"
    prov_record = _prov_payload(source_name, payload)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prov_record, ensure_ascii=True) + "\n")
    return path
