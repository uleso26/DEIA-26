from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MONGO_DIR = PROCESSED_DIR / "mongo"
CANONICAL_DIR = DATA_DIR / "canonical"
LOG_DIR = ROOT / "logs" / "query_traces"
LINEAGE_DIR = ROOT / "logs" / "ingestion_lineage"
SQLITE_DB = PROCESSED_DIR / "t2d_platform.db"
GRAPH_FILE = PROCESSED_DIR / "neo4j_graph.json"
RETRIEVAL_MANIFEST = PROCESSED_DIR / "retrieval_manifest.json"
# Backward-compatible alias for older code paths that still import FAISS_MANIFEST.
FAISS_MANIFEST = RETRIEVAL_MANIFEST


def ensure_runtime_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MONGO_DIR, LOG_DIR, LINEAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def relative_runtime_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)
