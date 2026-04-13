# Import the libraries helpers and shared models needed in this file
from pathlib import Path


# Define the constants lookup tables and settings used below
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


# Ensure the runtime directory structure exists before files are written
def ensure_runtime_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR, MONGO_DIR, LOG_DIR, LINEAGE_DIR, PROV_LINEAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)


# Return the resolved path for a raw input file
def raw_input_path(filename: str) -> Path:
    """Read from runtime/raw when present, otherwise fall back to tracked fixtures."""
    runtime_path = RAW_DIR / filename
    if runtime_path.exists():
        return runtime_path
    return FIXTURE_RAW_DIR / filename


# Return a runtime path relative to the repository root
def relative_runtime_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)
