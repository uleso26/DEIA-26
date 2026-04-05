from __future__ import annotations

import json
import re
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from core.paths import LINEAGE_DIR, PROV_LINEAGE_DIR, RETRIEVAL_MANIFEST
from core.storage import (
    backend_status,
    build_chroma_index,
    build_dense_index,
    connect_sqlite,
    search_dense_index,
)
from data.ingestion.base import append_prov_manifest, validate_records
from data.ingestion.ingest_pubmed import _summary_matches_seed
from tools.native_tools import filter_recent_documents


pytestmark = pytest.mark.integration


def test_auxiliary_source_tables_are_populated() -> None:
    connection = connect_sqlite()
    try:
        cursor = connection.cursor()
        counts: dict[str, int] = {}
        allowed_tables = {
            "who_gho_diabetes_stats",
            "drugbank_classifications",
            "synthetic_patient_profiles",
        }
        requested_tables = (
            "who_gho_diabetes_stats",
            "drugbank_classifications",
            "synthetic_patient_profiles",
        )
        for table in requested_tables:
            assert table in allowed_tables
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
    finally:
        connection.close()
    for count in counts.values():
        assert count >= 1


def test_backend_status_reports_expected_shape() -> None:
    status = backend_status()
    assert "sqlite" in status
    assert "mongodb" in status
    assert "neo4j" in status
    assert "fallback_files" in status
    assert status["sqlite"]["available"]
    assert "retrieval_manifest" in status["fallback_files"]
    assert "chroma_dir" in status["fallback_files"]


def test_retrieval_manifest_and_lineage_artifacts_exist() -> None:
    assert RETRIEVAL_MANIFEST.exists()
    for source_name in ["openfda", "clinicaltrials", "pubmed"]:
        assert (LINEAGE_DIR / f"{source_name}.jsonl").exists()
        assert (PROV_LINEAGE_DIR / f"{source_name}.jsonl").exists()


def test_pubmed_live_guard_rejects_semantic_mismatch() -> None:
    seed_document = {
        "title": "SGLT2 inhibitors and heart failure outcomes in type 2 diabetes",
        "mesh_terms": ["SGLT2 inhibitors", "heart failure", "type 2 diabetes"],
        "text": "Cardiorenal benefit was observed in type 2 diabetes populations.",
    }
    mismatched_summary = {
        "title": "Early Periprosthetic Tibial Lucency Following Low-Profile Total Ankle Arthroplasty.",
        "fulljournalname": "Foot & ankle international",
    }
    assert not _summary_matches_seed(seed_document, mismatched_summary)


def test_filter_recent_documents_skips_malformed_dates() -> None:
    documents = [
        {"publication_date": "2026-03-01", "title": "valid"},
        {"publication_date": "not-a-date", "title": "broken"},
    ]
    filtered = filter_recent_documents(documents, months=6)
    assert [item["title"] for item in filtered] == ["valid"]


def test_filter_recent_documents_honors_reference_date() -> None:
    documents = [
        {"publication_date": "2025-10-02", "title": "inside"},
        {"publication_date": "2025-10-01", "title": "outside"},
    ]
    filtered = filter_recent_documents(documents, months=6, reference_date=date(2026, 4, 2))
    assert [item["title"] for item in filtered] == ["inside"]


def test_build_dense_index_and_chroma_search_return_expected_match() -> None:
    documents = [
        {"title": "GLP1R therapy", "text": "GLP1R incretin pathway evidence."},
        {"title": "SGLT2 outcomes", "text": "SGLT2 heart failure evidence."},
    ]

    def fake_embed(
        texts: list[str],
        *,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> tuple[list[list[float]], dict[str, str]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "glp1r" in lowered:
                vectors.append([1.0, 0.0])
            elif "sglt2" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors, {"embedding_provider": "stub", "embedding_model": "stub-model"}

    collection_name = f"test-chroma-{uuid.uuid4().hex[:8]}"
    with patch("core.storage._embed_texts", side_effect=fake_embed):
        dense_manifest = build_dense_index(documents)
        chroma_manifest = build_chroma_index(documents, collection_name=collection_name)
        assert dense_manifest is not None
        assert dense_manifest["vector_dim"] == 2
        assert chroma_manifest is not None
        results = search_dense_index(
            "GLP1R therapy",
            {
                "chroma_collection": chroma_manifest["collection_name"],
                "embedding_provider": chroma_manifest["embedding_provider"],
                "embedding_model": chroma_manifest["embedding_model"],
            },
            top_k=1,
        )
    assert results[0]["title"] == "GLP1R therapy"


def test_append_prov_manifest_writes_valid_prov_record() -> None:
    source_name = f"unit_test_prov_{uuid.uuid4().hex[:8]}"
    path = append_prov_manifest(
        source_name,
        {
            "recorded_at": "2026-04-02T12:00:00+00:00",
            "mode": "seed_fixture",
            "raw_files": {"pubmed": "runtime/raw/pubmed_documents.json"},
            "record_counts": {"pubmed": 2},
            "upstream_requests": [{"url": "https://example.org/pubmed", "status_code": 200, "ok": True}],
        },
    )
    record = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["prefix"]["prov"] == "http://www.w3.org/ns/prov#"
    assert record["activity"]["mode"] == "seed_fixture"
    assert record["generated"][0]["location"] == "runtime/raw/pubmed_documents.json"
    assert record["used"][0]["status_code"] == 200


def test_validate_records_filters_missing_required_fields() -> None:
    records = [
        {"title": "valid", "text": "ok"},
        {"title": "missing text"},
    ]
    filtered = validate_records(records, ["title", "text"], "unit_test")
    assert filtered == [{"title": "valid", "text": "ok"}]


def test_repo_does_not_ship_real_secrets_or_absolute_local_paths() -> None:
    sensitive_files = [
        Path("README.md"),
        Path(".env.example"),
        Path("SECURITY.md"),
    ]
    secret_pattern = re.compile(r"sk-[A-Za-z0-9]{20,}")
    for path in sensitive_files:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, f"Absolute local path leaked in {path}"
        assert secret_pattern.search(text) is None, f"Secret-like token leaked in {path}"
