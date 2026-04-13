# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import argparse

from data.ingestion.base import write_seed_payload
from data.ingestion.seed_data import SYNTHETIC_PATIENT_PROFILES


# Run the main workflow implemented by this module
def run() -> str:
    return write_seed_payload(
        "synthetic_patient_profiles",
        "synthetic_patient_profiles.json",
        SYNTHETIC_PATIENT_PROFILES,
    )


# Coordinate the main execution path for this module
def main() -> None:
    parser = argparse.ArgumentParser(description="Write seed synthetic medical profiles.")
    parser.parse_args()
    print(run())


# CLI entrypoint
if __name__ == "__main__":
    main()
