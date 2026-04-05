from __future__ import annotations

import argparse

from data.ingestion.base import write_seed_payload
from data.ingestion.seed_data import EXTERNAL_INTELLIGENCE


def run() -> str:
    return write_seed_payload(
        "external_intelligence",
        "external_intelligence.json",
        EXTERNAL_INTELLIGENCE,
        timestamp_fields=["scraped_at"],
        required_fields=["headline", "summary", "publish_date", "source_url"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed external intelligence payload.")
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
