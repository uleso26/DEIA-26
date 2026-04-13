# Imports.
from __future__ import annotations

import json
import math
import os
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.paths import CHROMA_DIR, GRAPH_FILE, MONGO_DIR, RETRIEVAL_MANIFEST, SQLITE_DB, ensure_runtime_directories, relative_runtime_path
from core.logging_utils import get_logger
from core.runtime_utils import env_flag


# Keep chunks short enough to stay semantically tight while still carrying a
# full trial result sentence or guideline recommendation.
DEFAULT_CHUNK_MAX_CHARS = 550
logger = get_logger(__name__)


# Load JSON.
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Dump JSON.
def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


# Connect SQLite.
def connect_sqlite() -> sqlite3.Connection:
    ensure_runtime_directories()
    return sqlite3.connect(SQLITE_DB)


# Use live MongoDB.
def use_live_mongodb(prefer_service: bool = False) -> bool:
    return prefer_service or env_flag("USE_MONGODB_BACKEND", False)


# Use live Neo4j.
def use_live_neo4j(prefer_service: bool = False) -> bool:
    return prefer_service or env_flag("USE_NEO4J_BACKEND", False)


# Get Mongo client.
@lru_cache(maxsize=4)
def _get_mongo_client(mongodb_uri: str) -> Any | None:
    try:
        from pymongo import MongoClient  # type: ignore
    except ImportError:
        return None
    return MongoClient(mongodb_uri, serverSelectionTimeoutMS=2000)


# Load collection from MongoDB.
def load_collection_from_mongodb(name: str) -> tuple[list[dict[str, Any]], bool]:
    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE", "t2d_intelligence")
    if not mongodb_uri:
        return [], False
    client = _get_mongo_client(mongodb_uri)
    if client is None:
        return [], False
    try:
        collection = client[database_name][name]
        documents = list(collection.find({}, {"_id": 0}))
        return documents, True
    except Exception as exc:
        logger.warning("MongoDB fallback for collection '%s': %s", name, exc)
        return [], False


# Load collection with backend.
def load_collection_with_backend(name: str, prefer_service: bool = False) -> tuple[list[dict[str, Any]], str]:
    if use_live_mongodb(prefer_service):
        documents, loaded = load_collection_from_mongodb(name)
        if loaded:
            return documents, "mongodb"
    path = MONGO_DIR / f"{name}.json"
    if not path.exists():
        return [], "mongo_fallback"
    return load_json(path), "mongo_fallback"


# Load collection.
def load_collection(name: str, prefer_service: bool = False) -> list[dict[str, Any]]:
    return load_collection_with_backend(name, prefer_service=prefer_service)[0]


# Run Neo4j query.
def _run_neo4j_query(query: str, parameters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], bool]:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not all([uri, user, password]):
        return [], False
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        return [], False

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result], True
    except Exception as exc:
        logger.warning("Neo4j fallback for query execution: %s", exc)
        return [], False
    finally:
        driver.close()


# Run Neo4j query with backend.
def run_neo4j_query_with_backend(
    query: str,
    parameters: dict[str, Any] | None = None,
    prefer_service: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    if use_live_neo4j(prefer_service):
        records, succeeded = _run_neo4j_query(query, parameters)
        if succeeded:
            return records, "neo4j"
    return [], "neo4j_fallback"


# Run Neo4j query.
def run_neo4j_query(query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return run_neo4j_query_with_backend(query, parameters)[0]


# Backend status.
def backend_status() -> dict[str, Any]:
    mongo_documents, mongo_backend = load_collection_with_backend("clinical_trials", prefer_service=True)
    neo4j_probe_records, neo4j_backend = run_neo4j_query_with_backend("RETURN 1 AS ok", prefer_service=True)
    retrieval_manifest = load_retrieval_manifest()
    neo4j_entity_records, _ = run_neo4j_query_with_backend(
        "MATCH (n:Entity) RETURN count(n) AS entity_count",
        prefer_service=True,
    )
    neo4j_relation_records, _ = run_neo4j_query_with_backend(
        "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) AS relation_count",
        prefer_service=True,
    )
    entity_count = neo4j_entity_records[0]["entity_count"] if neo4j_entity_records else 0
    relation_count = neo4j_relation_records[0]["relation_count"] if neo4j_relation_records else 0
    return {
        "sqlite": {
            "path": relative_runtime_path(SQLITE_DB),
            "available": SQLITE_DB.exists(),
        },
        "mongodb": {
            "configured": bool(os.getenv("MONGODB_URI")),
            "backend": mongo_backend,
            "available": mongo_backend == "mongodb",
            "sample_collection_count": len(mongo_documents),
        },
        "neo4j": {
            "configured": all(bool(os.getenv(name)) for name in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]),
            "backend": neo4j_backend,
            "available": neo4j_backend == "neo4j" and bool(neo4j_probe_records),
            "probe_records": len(neo4j_probe_records),
            "entity_count": entity_count,
            "relation_count": relation_count,
            "seeded": entity_count > 0 and relation_count > 0,
        },
        "fallback_files": {
            "mongo_dir": relative_runtime_path(MONGO_DIR),
            "graph_file": relative_runtime_path(GRAPH_FILE),
            "retrieval_manifest": relative_runtime_path(RETRIEVAL_MANIFEST),
            "chroma_dir": relative_runtime_path(CHROMA_DIR),
            "graph_available": GRAPH_FILE.exists(),
            "retrieval_available": RETRIEVAL_MANIFEST.exists(),
            "retrieval_backend": retrieval_manifest.get("backend", "lexical"),
            "embedding_provider": retrieval_manifest.get("embedding_provider"),
            "embedding_model": retrieval_manifest.get("embedding_model"),
            "chroma_collection": retrieval_manifest.get("chroma_collection"),
        },
    }


# Load graph.
def load_graph() -> dict[str, Any]:
    if not GRAPH_FILE.exists():
        return {"nodes": [], "edges": []}
    return load_json(GRAPH_FILE)


# Load retrieval manifest.
def load_retrieval_manifest() -> dict[str, Any]:
    if not RETRIEVAL_MANIFEST.exists():
        return {"documents": [], "idf": {}, "doc_vectors": {}, "dense_vectors": {}}
    return load_json(RETRIEVAL_MANIFEST)


# Tokenize.
def tokenize(text: str) -> list[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [token for token in cleaned.split() if token]


# Embed texts.
def _embed_texts(
    texts: list[str],
    *,
    provider: str | None = None,
    model_name: str | None = None,
) -> tuple[list[list[float]] | None, dict[str, str]]:
    # Keep embedding dependencies off the import path for normal storage reads.
    from core.embeddings import embed_texts

    return embed_texts(texts, provider=provider, model_name=model_name)


# Build lexical index.
def build_lexical_index(documents: list[dict[str, Any]], text_key: str = "text") -> dict[str, Any]:
    doc_term_counts: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    stored_documents: list[dict[str, Any]] = []

    for index, document in enumerate(documents):
        doc_id = str(document.get("doc_id") or document.get("chunk_id") or document.get("id") or document.get("pmid") or index)
        text = document.get(text_key, "")
        terms = Counter(tokenize(text))
        for term in terms:
            document_frequency[term] += 1
        doc_term_counts[doc_id] = terms
        stored = dict(document)
        stored["doc_id"] = doc_id
        stored_documents.append(stored)

    doc_count = max(len(stored_documents), 1)
    idf = {
        term: math.log((1 + doc_count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }
    doc_vectors = {
        doc_id: {term: count * idf.get(term, 1.0) for term, count in counts.items()}
        for doc_id, counts in doc_term_counts.items()
    }
    return {"documents": stored_documents, "idf": idf, "doc_vectors": doc_vectors}


# Chunk retrieval documents.
def chunk_retrieval_documents(
    documents: list[dict[str, Any]],
    *,
    text_key: str = "text",
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Split long retrieval documents into sentence-aware chunks for indexing."""
    chunked_documents: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        base_id = str(document.get("id") or document.get("pmid") or document.get("doc_id") or index)
        text = str(document.get(text_key, "")).strip()
        if not text:
            continue
        sentences = [sentence.strip() for sentence in text.replace("\n", " ").split(". ") if sentence.strip()]
        if not sentences:
            sentences = [text]

        chunk_buffer: list[str] = []
        chunk_index = 0
        for sentence in sentences:
            candidate = ". ".join([*chunk_buffer, sentence]).strip()
            if chunk_buffer and len(candidate) > max_chars:
                chunk_text = ". ".join(chunk_buffer).strip()
                chunked_documents.append(
                    {
                        **document,
                        "parent_doc_id": base_id,
                        "chunk_id": f"{base_id}::chunk::{chunk_index}",
                        "doc_id": f"{base_id}::chunk::{chunk_index}",
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text if chunk_text.endswith(".") else f"{chunk_text}.",
                    }
                )
                chunk_index += 1
                chunk_buffer = [sentence]
            else:
                chunk_buffer.append(sentence)

        if chunk_buffer:
            chunk_text = ". ".join(chunk_buffer).strip()
            chunked_documents.append(
                {
                    **document,
                    "parent_doc_id": base_id,
                    "chunk_id": f"{base_id}::chunk::{chunk_index}",
                    "doc_id": f"{base_id}::chunk::{chunk_index}",
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text if chunk_text.endswith(".") else f"{chunk_text}.",
                }
            )
    return chunked_documents


# Build dense index.
def build_dense_index(documents: list[dict[str, Any]], text_key: str = "text") -> dict[str, Any] | None:
    """Build an in-memory dense vector index payload for small local runs."""
    stored_documents: list[dict[str, Any]] = []
    texts: list[str] = []
    for index, document in enumerate(documents):
        doc_id = str(document.get("doc_id") or document.get("chunk_id") or document.get("id") or document.get("pmid") or index)
        stored = dict(document)
        stored["doc_id"] = doc_id
        stored_documents.append(stored)
        texts.append(str(document.get(text_key, "")))

    vectors, metadata = _embed_texts(texts)
    if not vectors or len(vectors) != len(stored_documents):
        logger.warning("Dense index build skipped because embeddings were unavailable or incomplete.")
        return None

    dense_vectors = {
        document["doc_id"]: [round(float(value), 8) for value in vector]
        for document, vector in zip(stored_documents, vectors, strict=False)
    }
    vector_dim = len(vectors[0]) if vectors else 0
    return {
        "documents": stored_documents,
        "dense_vectors": dense_vectors,
        "vector_dim": vector_dim,
        **metadata,
    }


# Build chroma index.
def build_chroma_index(
    documents: list[dict[str, Any]],
    text_key: str = "text",
    collection_name: str = "t2d_retrieval_chunks",
) -> dict[str, Any] | None:
    """Persist chunk embeddings into a local Chroma collection when available."""
    stored_documents: list[dict[str, Any]] = []
    texts: list[str] = []
    for index, document in enumerate(documents):
        doc_id = str(document.get("doc_id") or document.get("chunk_id") or document.get("id") or document.get("pmid") or index)
        stored = dict(document)
        stored["doc_id"] = doc_id
        stored_documents.append(stored)
        texts.append(str(document.get(text_key, "")))

    vectors, metadata = _embed_texts(texts)
    if not vectors or len(vectors) != len(stored_documents):
        logger.warning("Chroma index build skipped because embeddings were unavailable or incomplete.")
        return None

    try:
        import chromadb
    except Exception as exc:
        logger.warning("Chroma backend unavailable; falling back to manifest-only retrieval: %s", exc)
        return None

    ensure_runtime_directories()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception as exc:
        logger.warning("Chroma collection reset skipped for '%s': %s", collection_name, exc)
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "domain": "t2d_retrieval",
            "hnsw:space": "cosine",
        },
    )
    metadatas = []
    ids = []
    documents_payload = []
    for document in stored_documents:
        ids.append(document["doc_id"])
        documents_payload.append(str(document.get(text_key, "")))
        metadatas.append(
            {
                "doc_id": document["doc_id"],
                "parent_doc_id": str(document.get("parent_doc_id", document["doc_id"])),
                "title": str(document.get("title", "")),
                "journal": str(document.get("journal", "")),
                "publication_date": str(document.get("publication_date", "")),
                "pmid": str(document.get("pmid", document.get("doc_id", ""))),
                "source_url": str(document.get("source_url", "")),
                "evidence_type": str(document.get("evidence_type", "")),
                "mesh_terms": json.dumps(document.get("mesh_terms", [])),
                "chunk_text": str(document.get(text_key, "")),
            }
        )
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents_payload)
    vector_dim = len(vectors[0]) if vectors else 0
    return {
        "collection_name": collection_name,
        "vector_dim": vector_dim,
        **metadata,
    }


# Search lexical index.
def search_lexical_index(query: str, manifest: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return []

    query_vector = {
        term: count * manifest.get("idf", {}).get(term, 1.0) for term, count in query_terms.items()
    }
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in manifest.get("documents", []):
        vector = manifest.get("doc_vectors", {}).get(document["doc_id"], {})
        score = sum(query_vector.get(term, 0.0) * vector.get(term, 0.0) for term in query_vector)
        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, Any]] = []
    for score, document in scored[:top_k]:
        item = dict(document)
        item["score"] = round(score, 4)
        results.append(item)
    return results


# Search dense index.
def search_dense_index(query: str, manifest: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    """Search the dense retrieval backend, preferring Chroma when configured."""
    collection_name = manifest.get("chroma_collection")
    if collection_name:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(collection_name)
            query_vectors, _ = _embed_texts(
                [query],
                provider=manifest.get("embedding_provider"),
                model_name=manifest.get("embedding_model"),
            )
            if not query_vectors:
                return []
            payload = collection.query(
                query_embeddings=query_vectors,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            ids = payload.get("ids", [[]])[0]
            metadatas = payload.get("metadatas", [[]])[0]
            documents_payload = payload.get("documents", [[]])[0]
            distances = payload.get("distances", [[]])[0]
            results: list[dict[str, Any]] = []
            for doc_id, metadata, chunk_text, distance in zip(ids, metadatas, documents_payload, distances):
                metadata = metadata or {}
                item = {
                    "doc_id": doc_id,
                    "parent_doc_id": metadata.get("parent_doc_id"),
                    "pmid": metadata.get("pmid"),
                    "title": metadata.get("title"),
                    "journal": metadata.get("journal"),
                    "publication_date": metadata.get("publication_date"),
                    "source_url": metadata.get("source_url"),
                    "evidence_type": metadata.get("evidence_type"),
                    "mesh_terms": json.loads(metadata.get("mesh_terms", "[]")),
                    "text": chunk_text,
                    "score": round(max(0.0, 1.0 - float(distance or 0.0)), 4),
                }
                results.append(item)
            return results
        except Exception as exc:
            logger.warning("Chroma dense search fallback for collection '%s': %s", collection_name, exc)
            return []

    dense_vectors = manifest.get("dense_vectors", {})
    documents = manifest.get("documents", [])
    if not dense_vectors or not documents:
        return []

    query_vectors, _ = _embed_texts(
        [query],
        provider=manifest.get("embedding_provider"),
        model_name=manifest.get("embedding_model"),
    )
    if not query_vectors:
        logger.warning("Dense search fallback because query embedding was unavailable.")
        return []
    query_vector = query_vectors[0]
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        vector = dense_vectors.get(document["doc_id"])
        if not vector:
            continue
        score = sum(float(left) * float(right) for left, right in zip(query_vector, vector))
        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, Any]] = []
    for score, document in scored[:top_k]:
        item = dict(document)
        item["score"] = round(score, 4)
        results.append(item)
    return results
