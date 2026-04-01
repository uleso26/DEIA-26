from __future__ import annotations

import argparse
import os
import re
from urllib.parse import quote

from core.paths import relative_runtime_path
from data.ingestion.base import append_lineage_manifest, fetch_json_response, live_ingestion_enabled, write_raw_payload
from data.ingestion.seed_data import PUBMED_DOCUMENTS


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4
    }


def _normalize_pubdate(raw_value: object, fallback: str) -> str:
    text = str(raw_value or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text
    if re.match(r"^\d{4}-\d{2}$", text):
        return f"{text}-01"
    if re.match(r"^\d{4}$", text):
        return f"{text}-01-01"
    return fallback


def _summary_matches_seed(seed_document: dict[str, object], summary: dict[str, object]) -> bool:
    seed_tokens = _tokenize(
        f"{seed_document['title']} {' '.join(seed_document['mesh_terms'])} {seed_document['text']}"
    )
    live_tokens = _tokenize(
        f"{summary.get('title', '')} {summary.get('fulljournalname', '')}"
    )
    overlap = seed_tokens.intersection(live_tokens)
    return len(overlap) >= 2


def _fetch_pubmed_summary(seed_document: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object]]:
    base_url = os.getenv("PUBMED_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils").rstrip("/")
    pmid = str(seed_document["pmid"])
    url = f"{base_url}/esummary.fcgi?db=pubmed&id={quote(pmid, safe='')}&retmode=json"
    response = fetch_json_response(url)
    payload = response.get("payload") if response.get("ok") else None
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    summary = result.get(pmid, {}) if isinstance(result, dict) else {}
    request_log = {
        "dataset": "pubmed",
        "url": url,
        "pmid": pmid,
        "status_code": response.get("status_code"),
        "ok": response.get("ok"),
        "record_count": 1 if summary else 0,
        "error": response.get("error"),
    }
    if not summary:
        return None, request_log
    if not _summary_matches_seed(seed_document, summary):
        request_log["accepted"] = False
        request_log["fallback_reason"] = "semantic_mismatch"
        return None, request_log
    request_log["accepted"] = True
    normalized = {
        "pmid": pmid,
        "title": summary.get("title", seed_document["title"]),
        "journal": summary.get("fulljournalname", seed_document["journal"]),
        "publication_date": _normalize_pubdate(summary.get("pubdate"), str(seed_document["publication_date"])),
        "evidence_type": seed_document["evidence_type"],
        "mesh_terms": list(seed_document["mesh_terms"]),
        "text": seed_document["text"],
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "data_mode": "live_metadata_plus_seed_text",
    }
    return normalized, request_log


def _seed_document(seed_document: dict[str, object]) -> dict[str, object]:
    normalized = dict(seed_document)
    normalized["data_mode"] = "seed_curated_literature"
    return normalized


def run() -> str:
    use_live = live_ingestion_enabled("pubmed")
    normalized_documents: list[dict[str, object]] = []
    request_log: list[dict[str, object]] = []
    accepted_live_records = 0

    if use_live:
        for seed_document in PUBMED_DOCUMENTS:
            live_document, request = _fetch_pubmed_summary(seed_document)
            request_log.append(request)
            if live_document:
                accepted_live_records += 1
            normalized_documents.append(live_document or _seed_document(seed_document))
    else:
        normalized_documents = [_seed_document(item) for item in PUBMED_DOCUMENTS]

    if not use_live:
        mode = "seed_fallback"
    elif accepted_live_records == len(PUBMED_DOCUMENTS):
        mode = "live_api"
    elif accepted_live_records > 0:
        mode = "partial_live_api"
    else:
        mode = "seed_fallback"
    if mode == "seed_fallback":
        normalized_documents = [_seed_document(item) for item in PUBMED_DOCUMENTS]

    path = write_raw_payload("pubmed_documents.json", normalized_documents)
    append_lineage_manifest(
        "pubmed",
        {
            "mode": mode,
            "raw_files": {"pubmed_documents": relative_runtime_path(path)},
            "record_counts": {"pubmed_documents": len(normalized_documents)},
            "accepted_live_records": accepted_live_records,
            "upstream_requests": request_log,
        },
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed PubMed payload.")
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
