# Imports.
from __future__ import annotations

import argparse

from data.ingestion.base import write_seed_payload
from data.ingestion.seed_data import GUIDELINE_EXCERPTS


# Run.
def run() -> str:
    return write_seed_payload(
        "guideline_excerpts",
        "guideline_excerpts.json",
        GUIDELINE_EXCERPTS,
        required_fields=["id", "title", "publication_date", "text", "source_url"],
    )


# Main.
def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed guideline excerpts payload.")
    parser.parse_args()
    print(run())


# CLI entrypoint.
if __name__ == "__main__":
    main()
