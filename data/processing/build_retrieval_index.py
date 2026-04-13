# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import argparse

from core.paths import RETRIEVAL_MANIFEST, raw_input_path
from core.storage import build_chroma_index, build_lexical_index, chunk_retrieval_documents, dump_json, load_json


# Enrich records with retrieval text for downstream use
def _with_retrieval_text(documents: list[dict]) -> list[dict]:
    enriched_documents = []
    for document in documents:
        enriched = dict(document)
        # Pull the searchable fields together once so the retrieval layer stays simple
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


# Run the main workflow implemented by this module
def run() -> str:
    documents = load_json(raw_input_path("pubmed_documents.json"))
    guideline_path = raw_input_path("guideline_excerpts.json")
    if guideline_path.exists():
        documents = [*documents, *load_json(guideline_path)]
    retrieval_documents = _with_retrieval_text(documents)
    chunked_documents = chunk_retrieval_documents(retrieval_documents, text_key="retrieval_text")
    manifest = build_lexical_index(chunked_documents, text_key="chunk_text")
    manifest["documents"] = chunked_documents
    manifest["chunking"] = {
        "strategy": "sentence_window",
        "text_key": "retrieval_text",
        "chunk_text_key": "chunk_text",
    }
    manifest["backend"] = "lexical"
    chroma_manifest = build_chroma_index(chunked_documents, text_key="chunk_text")
    if chroma_manifest:
        manifest["chroma_collection"] = chroma_manifest["collection_name"]
        manifest["embedding_provider"] = chroma_manifest["embedding_provider"]
        manifest["embedding_model"] = chroma_manifest["embedding_model"]
        manifest["vector_dim"] = chroma_manifest["vector_dim"]
        manifest["backend"] = "hybrid_chroma"
    else:
        manifest["dense_vectors"] = {}
    dump_json(RETRIEVAL_MANIFEST, manifest)
    return str(RETRIEVAL_MANIFEST)


# Coordinate the main execution path for this module
def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval manifest with lexical fallback indexing.")
    parser.parse_args()
    print(run())


# CLI entrypoint
if __name__ == "__main__":
    main()
