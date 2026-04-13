# Imports.
from __future__ import annotations

import argparse
import os
import time

from core.paths import MONGO_DIR, raw_input_path
from core.storage import dump_json, load_json
from data.ingestion.seed_data import EXTERNAL_INTELLIGENCE


# Sync to MongoDB.
def _sync_to_mongodb(collections: dict[str, list[dict]]) -> None:
    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE", "t2d_intelligence")
    if not mongodb_uri:
        return
    try:
        from pymongo import MongoClient  # type: ignore
    except ImportError:
        return

    retries = int(os.getenv("MONGODB_SYNC_RETRIES", "10"))
    delay_seconds = float(os.getenv("MONGODB_SYNC_DELAY_SECONDS", "1.5"))

    for _ in range(retries):
        client = None
        try:
            client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
            )
            client.admin.command("ping")
            database = client[database_name]
            for name, documents in collections.items():
                collection = database[name]
                collection.delete_many({})
                if documents:
                    collection.insert_many(documents)
            return
        except Exception:
            time.sleep(delay_seconds)
        finally:
            if client is not None:
                client.close()


# Run.
def run(sync: bool = False) -> dict[str, str]:
    MONGO_DIR.mkdir(parents=True, exist_ok=True)
    collections = {
        "clinical_trials": load_json(raw_input_path("clinical_trials.json")),
        "external_intelligence": load_json(raw_input_path("external_intelligence.json"))
        if raw_input_path("external_intelligence.json").exists()
        else EXTERNAL_INTELLIGENCE,
    }
    for name, payload in collections.items():
        dump_json(MONGO_DIR / f"{name}.json", payload)
    if sync:
        _sync_to_mongodb(collections)
    return {name: str(MONGO_DIR / f"{name}.json") for name in collections}


# Main.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mongo-style collection artifacts.")
    parser.add_argument("--sync", action="store_true", help="Attempt to sync to MongoDB if pymongo is available.")
    args = parser.parse_args()
    print(run(sync=args.sync))


# CLI entrypoint.
if __name__ == "__main__":
    main()
