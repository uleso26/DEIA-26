"""Hybrid retrieval and lightweight PubMed search helpers."""

from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Any

from core.paths import raw_input_path
from core.storage import (
    build_chroma_index,
    build_lexical_index,
    chunk_retrieval_documents,
    load_json,
    load_retrieval_manifest,
    search_dense_index,
    search_lexical_index,
    tokenize,
)
from data.canonical.resolver import get_resolver


# Hybrid retrieval leans slightly toward dense search because the corpus is
# small and terminology can vary, but lexical overlap still helps anchor exact
# trial names and guideline phrasing.
LEXICAL_FUSION_WEIGHT = 0.4
DENSE_FUSION_WEIGHT = 0.6

# Domain reranking heuristics keep obvious entity and intent hits at the top
# after lexical+dense fusion has narrowed the field.
DRUG_MATCH_BOOST = 6.0
TARGET_MATCH_BOOST = 4.0
SAFETY_INTENT_BOOST = 3.0
POST_MARKETING_BOOST = 2.0
SURVEILLANCE_BOOST = 1.5
HEART_FAILURE_BOOST = 3.0


def _pubmed_documents() -> list[dict[str, Any]]:
    return load_json(raw_input_path("pubmed_documents.json"))


def _retrieval_documents() -> list[dict[str, Any]]:
    documents = []
    source_documents = [*_pubmed_documents()]
    guideline_path = raw_input_path("guideline_excerpts.json")
    if guideline_path.exists():
        source_documents.extend(load_json(guideline_path))
    for document in source_documents:
        enriched = dict(document)
        enriched["retrieval_text"] = " ".join(
            [
                str(document.get("title", "")),
                str(document.get("journal", "")),
                " ".join(document.get("mesh_terms", [])),
                str(document.get("text", "")),
            ]
        ).strip()
        documents.append(enriched)
    return documents


def search_retrieval_index(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Run hybrid retrieval over chunked literature and guideline evidence."""
    manifest = load_retrieval_manifest()
    if not manifest.get("documents"):
        fallback_documents = chunk_retrieval_documents(_retrieval_documents(), text_key="retrieval_text")
        manifest = build_lexical_index(fallback_documents, text_key="chunk_text")
        manifest["documents"] = fallback_documents
        manifest["backend"] = "lexical"
        chroma_manifest = build_chroma_index(fallback_documents, text_key="chunk_text")
        if chroma_manifest:
            manifest["chroma_collection"] = chroma_manifest["collection_name"]
            manifest["embedding_provider"] = chroma_manifest["embedding_provider"]
            manifest["embedding_model"] = chroma_manifest["embedding_model"]
            manifest["vector_dim"] = chroma_manifest["vector_dim"]
            manifest["backend"] = "hybrid_chroma"
        else:
            manifest["dense_vectors"] = {}
    candidate_count = max(top_k * 4, top_k)
    lexical_results = search_lexical_index(query, manifest, top_k=candidate_count)
    dense_results = search_dense_index(query, manifest, top_k=candidate_count)
    normalized_scores: dict[str, float] = {}
    for weight, results in [
        (LEXICAL_FUSION_WEIGHT, lexical_results),
        (DENSE_FUSION_WEIGHT, dense_results),
    ]:
        if not results:
            continue
        max_score = max(float(item.get("score", 0.0)) for item in results) or 1.0
        for item in results:
            doc_id = str(item.get("parent_doc_id") or item.get("doc_id") or item.get("pmid") or item.get("id"))
            normalized_scores[doc_id] = normalized_scores.get(doc_id, 0.0) + weight * (float(item.get("score", 0.0)) / max_score)

    initial_results = []
    seen_doc_ids: set[str] = set()
    for item in [*dense_results, *lexical_results]:
        doc_id = str(item.get("parent_doc_id") or item.get("doc_id") or item.get("pmid") or item.get("id"))
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        seeded = dict(item)
        seeded["score"] = round(normalized_scores.get(doc_id, float(item.get("score", 0.0))), 4)
        if seeded.get("chunk_text") and not seeded.get("text"):
            seeded["text"] = seeded["chunk_text"]
        initial_results.append(seeded)

    if not initial_results:
        initial_results = lexical_results
    resolver = get_resolver()
    matched_drugs = {item["canonical_id"] for item in resolver.find_drugs(query)}
    matched_target = (resolver.resolve_target(query) or {}).get("canonical_id")
    lowered_query = query.lower()

    reranked = []
    for item in initial_results:
        haystack = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("journal", "")),
                " ".join(item.get("mesh_terms", [])),
                str(item.get("text", "")),
            ]
        ).lower()
        rerank_score = float(item.get("score", 0.0))
        for drug in matched_drugs:
            if drug.lower() in haystack:
                rerank_score += DRUG_MATCH_BOOST
        if matched_target and matched_target.lower() in haystack:
            rerank_score += TARGET_MATCH_BOOST
        if "safety" in lowered_query and any(term in haystack for term in ["safety", "faers", "pharmacovigilance"]):
            rerank_score += SAFETY_INTENT_BOOST
        if "post-marketing" in lowered_query and "post-marketing" in haystack:
            rerank_score += POST_MARKETING_BOOST
        if "surveillance" in lowered_query and any(term in haystack for term in ["surveillance", "pharmacovigilance"]):
            rerank_score += SURVEILLANCE_BOOST
        if "heart failure" in lowered_query and "heart failure" in haystack:
            rerank_score += HEART_FAILURE_BOOST
        seeded = dict(item)
        seeded["score"] = round(rerank_score, 4)
        reranked.append(seeded)

    reranked.sort(key=lambda item: (item.get("score", 0.0), item.get("publication_date", "")), reverse=True)
    return reranked[:top_k]


def search_pubmed(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Run a simple keyword match over the curated PubMed corpus."""
    documents = _pubmed_documents()
    query_tokens = tokenize(query)
    scored = []
    for document in documents:
        haystack_tokens = set(tokenize(f"{document['title']} {document['text']} {' '.join(document['mesh_terms'])}"))
        score = sum(1 for token in query_tokens if token in haystack_tokens)
        if score:
            item = dict(document)
            item["score"] = score
            scored.append((score, item))
    scored.sort(key=lambda item: (item[0], item[1]["publication_date"]), reverse=True)
    return [item for _, item in scored[:top_k]]


def search_pubmed_safety(drug: str, top_k: int = 2) -> list[dict[str, Any]]:
    """Search safety-oriented PubMed records for a therapy."""
    return search_pubmed(f"{drug} safety faers adverse", top_k=top_k)


def get_guideline_context(query: str) -> list[dict[str, Any]]:
    """Fetch guideline-flavoured literature context for a free-form query."""
    return search_pubmed(f"{query} guideline NICE ADA CKD", top_k=2)


def filter_recent_documents(documents: list[dict[str, Any]], months: int = 6, reference_date: date | None = None) -> list[dict[str, Any]]:
    """Keep documents within a recent-months window, skipping malformed dates."""
    today = reference_date or date.today()
    threshold_month = today.month - months
    threshold_year = today.year
    while threshold_month <= 0:
        threshold_month += 12
        threshold_year -= 1
    threshold_day = min(today.day, monthrange(threshold_year, threshold_month)[1])
    threshold = date(threshold_year, threshold_month, threshold_day)
    recent = []
    for document in documents:
        try:
            published = date.fromisoformat(str(document["publication_date"]))
        except (TypeError, ValueError):
            continue
        if published >= threshold:
            recent.append(document)
    return recent


__all__ = [
    "filter_recent_documents",
    "get_guideline_context",
    "search_pubmed",
    "search_pubmed_safety",
    "search_retrieval_index",
]
