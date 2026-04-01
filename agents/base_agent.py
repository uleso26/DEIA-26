from __future__ import annotations

from typing import Any

from core.models import Citation


def unique_strings(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def citation(source: str, title: str, reference_id: str, url: str, evidence_tier: str, published_at: str | None = None) -> Citation:
    return Citation(
        source=source,
        title=title,
        reference_id=reference_id,
        url=url,
        evidence_tier=evidence_tier,
        published_at=published_at,
    )


def dedupe_citations(citations: list[Citation]) -> list[Citation]:
    deduped: dict[str, Citation] = {}
    for item in citations:
        deduped[item.reference_id] = item
    return list(deduped.values())


def latest_record(records: list[dict[str, Any]], date_key: str) -> dict[str, Any] | None:
    if not records:
        return None
    return sorted(records, key=lambda record: record.get(date_key, ""), reverse=True)[0]
