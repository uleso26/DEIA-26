from __future__ import annotations

# Backward-compatible wrapper for older entrypoints. The active builder writes
# the retrieval manifest and prefers dense vectors when an embedding backend is available.
from data.processing.build_retrieval_index import main, run


if __name__ == "__main__":
    main()
