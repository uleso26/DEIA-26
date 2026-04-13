# Imports.
from pathlib import Path


# Module constants.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIXTURE_RAW_DIR = DATA_DIR / "raw"
RUNTIME_DIR = ROOT / "runtime"
RAW_DIR = RUNTIME_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = PROCESSED_DIR / "chroma"
MONGO_DIR = PROCESSED_DIR / "mongo"
CANONICAL_DIR = DATA_DIR / "canonical"
LOG_DIR = ROOT / "logs" / "query_traces"
LINEAGE_DIR = ROOT / "logs" / "ingestion_lineage"
PROV_LINEAGE_DIR = LINEAGE_DIR / "prov"
SQLITE_DB = PROCESSED_DIR / "t2d_platform.db"
GRAPH_FILE = PROCESSED_DIR / "neo4j_graph.json"
RETRIEVAL_MANIFEST = PROCESSED_DIR / "retrieval_manifest.json"


# Ensure runtime directories.
def ensure_runtime_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR, MONGO_DIR, LOG_DIR, LINEAGE_DIR, PROV_LINEAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)


# Raw input path.
def raw_input_path(filename: str) -> Path:
    """Read from runtime/raw when present, otherwise fall back to tracked fixtures."""
    runtime_path = RAW_DIR / filename
    if runtime_path.exists():
        return runtime_path
    return FIXTURE_RAW_DIR / filename


# Relative runtime path.
def relative_runtime_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)
