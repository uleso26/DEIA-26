from __future__ import annotations

from data.ingestion.ingest_drugbank_open import main, run


# Backward-compatible wrapper for the renamed DrugBank Open ingester.


if __name__ == "__main__":
    main()
