from __future__ import annotations

import argparse

from core.paths import RAW_DIR, RETRIEVAL_MANIFEST
from core.storage import build_dense_index, build_lexical_index, dump_json, load_json


def _with_retrieval_text(documents: list[dict]) -> list[dict]:
    enriched_documents = []
    for document in documents:
        enriched = dict(document)
        # Pull the searchable fields together once so the retrieval layer stays simple.
        enriched["retrieval_text"] = " ".join(
            [
                str(document.get("title", "")),
                str(document.get("journal", "")),
                " ".join(document.get("mesh_terms", [])),
                str(document.get("text", "")),
            ]
        ).strip()
        enriched_documents.append(enriched)
    return enriched_documents


def run() -> str:
    documents = load_json(RAW_DIR / "pubmed_documents.json")
    if (RAW_DIR / "guideline_excerpts.json").exists():
        documents = [*documents, *load_json(RAW_DIR / "guideline_excerpts.json")]
    retrieval_documents = _with_retrieval_text(documents)
    manifest = build_lexical_index(retrieval_documents, text_key="retrieval_text")
    dense_manifest = build_dense_index(retrieval_documents, text_key="retrieval_text")
    manifest["backend"] = "lexical"
    if dense_manifest:
        manifest["dense_vectors"] = dense_manifest["dense_vectors"]
        manifest["embedding_provider"] = dense_manifest["embedding_provider"]
        manifest["embedding_model"] = dense_manifest["embedding_model"]
        manifest["vector_dim"] = dense_manifest["vector_dim"]
        manifest["backend"] = "dense_vector"
    else:
        manifest["dense_vectors"] = {}
    dump_json(RETRIEVAL_MANIFEST, manifest)
    return str(RETRIEVAL_MANIFEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval manifest with lexical fallback indexing.")
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
