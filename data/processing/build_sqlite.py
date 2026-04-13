# Imports.
from __future__ import annotations

import argparse

from core.paths import CANONICAL_DIR, raw_input_path
from core.storage import connect_sqlite, load_json


# Module constants.
DDL = [
    """
    CREATE TABLE IF NOT EXISTS drug_labels (
        canonical_drug TEXT PRIMARY KEY,
        brand_names TEXT NOT NULL,
        label_version TEXT NOT NULL,
        indications TEXT NOT NULL,
        warnings TEXT NOT NULL,
        source_url TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS adverse_events (
        canonical_drug TEXT NOT NULL,
        event TEXT NOT NULL,
        count INTEGER NOT NULL,
        subgroup TEXT NOT NULL,
        time_period TEXT NOT NULL,
        signal_strength TEXT NOT NULL,
        source TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS drug_synonyms (
        canonical_drug TEXT NOT NULL,
        alias TEXT NOT NULL,
        drug_class TEXT NOT NULL,
        evidence_tier TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS target_synonyms (
        canonical_target TEXT NOT NULL,
        alias TEXT NOT NULL,
        evidence_tier TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trial_crosswalk (
        nct_id TEXT NOT NULL,
        trial_name TEXT NOT NULL,
        alias TEXT NOT NULL,
        evidence_tier TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_confidence_tiers (
        tier TEXT PRIMARY KEY,
        description TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS who_gho_diabetes_stats (
        country TEXT NOT NULL,
        indicator TEXT NOT NULL,
        year INTEGER NOT NULL,
        value REAL NOT NULL,
        unit TEXT NOT NULL,
        source_url TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS drugbank_classifications (
        canonical_drug TEXT NOT NULL,
        drugbank_id TEXT NOT NULL,
        drug_class TEXT NOT NULL,
        pharmacology TEXT NOT NULL,
        interactions TEXT NOT NULL,
        source_url TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS synthetic_patient_profiles (
        patient_id TEXT PRIMARY KEY,
        age_band TEXT NOT NULL,
        sex TEXT NOT NULL,
        bmi_category TEXT NOT NULL,
        ckd_stage TEXT NOT NULL,
        heart_failure INTEGER NOT NULL,
        current_therapy TEXT NOT NULL,
        clinical_priority TEXT NOT NULL,
        data_origin TEXT NOT NULL
    )
    """,
]


# Insert drug labels.
def _insert_drug_labels(cursor) -> None:
    labels = load_json(raw_input_path("drug_labels.json"))
    cursor.execute("DELETE FROM drug_labels")
    cursor.executemany(
        """
        INSERT INTO drug_labels (
            canonical_drug, brand_names, label_version, indications, warnings, source_url
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item["canonical_drug"],
                ",".join(item["brand_names"]),
                item["label_version"],
                ",".join(item["indications"]),
                ",".join(item["warnings"]),
                item["source_url"],
            )
            for item in labels
        ],
    )


# Insert adverse events.
def _insert_adverse_events(cursor) -> None:
    events = load_json(raw_input_path("openfda_adverse_events.json"))
    cursor.execute("DELETE FROM adverse_events")
    cursor.executemany(
        """
        INSERT INTO adverse_events (
            canonical_drug, event, count, subgroup, time_period, signal_strength, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item["canonical_drug"],
                item["event"],
                item["count"],
                item["subgroup"],
                item["time_period"],
                item["signal_strength"],
                item["source"],
            )
            for item in events
        ],
    )


# Insert canonical tables.
def _insert_canonical_tables(cursor) -> None:
    cursor.execute("DELETE FROM drug_synonyms")
    drug_synonyms = load_json(CANONICAL_DIR / "drug_synonyms.json")
    for canonical_drug, payload in drug_synonyms.items():
        aliases = sorted(set([canonical_drug, *payload["aliases"]]))
        cursor.executemany(
            "INSERT INTO drug_synonyms (canonical_drug, alias, drug_class, evidence_tier) VALUES (?, ?, ?, ?)",
            [
                (
                    canonical_drug,
                    alias,
                    payload["drug_class"],
                    payload["evidence_tier"],
                )
                for alias in aliases
            ],
        )

    cursor.execute("DELETE FROM target_synonyms")
    target_synonyms = load_json(CANONICAL_DIR / "target_synonyms.json")
    for canonical_target, payload in target_synonyms.items():
        aliases = sorted(set([canonical_target, *payload["aliases"]]))
        cursor.executemany(
            "INSERT INTO target_synonyms (canonical_target, alias, evidence_tier) VALUES (?, ?, ?)",
            [(canonical_target, alias, payload["evidence_tier"]) for alias in aliases],
        )

    cursor.execute("DELETE FROM trial_crosswalk")
    trial_crosswalk = load_json(CANONICAL_DIR / "trial_crosswalk.json")
    for nct_id, payload in trial_crosswalk.items():
        aliases = sorted(set([nct_id, payload["trial_name"], *payload["aliases"]]))
        cursor.executemany(
            "INSERT INTO trial_crosswalk (nct_id, trial_name, alias, evidence_tier) VALUES (?, ?, ?, ?)",
            [(nct_id, payload["trial_name"], alias, payload["evidence_tier"]) for alias in aliases],
        )

    cursor.execute("DELETE FROM entity_confidence_tiers")
    evidence_tiers = load_json(CANONICAL_DIR / "evidence_tiers.json")
    cursor.executemany(
        "INSERT INTO entity_confidence_tiers (tier, description) VALUES (?, ?)",
        list(evidence_tiers.items()),
    )


# Insert auxiliary reference tables.
def _insert_auxiliary_reference_tables(cursor) -> None:
    cursor.execute("DELETE FROM who_gho_diabetes_stats")
    who_path = raw_input_path("who_gho.json")
    if who_path.exists():
        who_data = load_json(who_path)
        cursor.executemany(
            """
            INSERT INTO who_gho_diabetes_stats (country, indicator, year, value, unit, source_url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["country"],
                    item["indicator"],
                    item["year"],
                    item["value"],
                    item["unit"],
                    item["source_url"],
                )
                for item in who_data
            ],
        )

    cursor.execute("DELETE FROM drugbank_classifications")
    drugbank_path = raw_input_path("drugbank_open.json")
    if drugbank_path.exists():
        drugbank_data = load_json(drugbank_path)
        cursor.executemany(
            """
            INSERT INTO drugbank_classifications (
                canonical_drug, drugbank_id, drug_class, pharmacology, interactions, source_url
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["canonical_drug"],
                    item["drugbank_id"],
                    item["drug_class"],
                    item["pharmacology"],
                    ",".join(item["interactions"]),
                    item["source_url"],
                )
                for item in drugbank_data
            ],
        )

    cursor.execute("DELETE FROM synthetic_patient_profiles")
    synthetic_path = raw_input_path("synthetic_patient_profiles.json")
    if synthetic_path.exists():
        synthetic_profiles = load_json(synthetic_path)
        cursor.executemany(
            """
            INSERT INTO synthetic_patient_profiles (
                patient_id, age_band, sex, bmi_category, ckd_stage, heart_failure,
                current_therapy, clinical_priority, data_origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["patient_id"],
                    item["age_band"],
                    item["sex"],
                    item["bmi_category"],
                    item["ckd_stage"],
                    int(bool(item["heart_failure"])),
                    item["current_therapy"],
                    item["clinical_priority"],
                    item["data_origin"],
                )
                for item in synthetic_profiles
            ],
        )


# Run.
def run() -> str:
    connection = connect_sqlite()
    try:
        cursor = connection.cursor()
        for statement in DDL:
            cursor.execute(statement)
        _insert_drug_labels(cursor)
        _insert_adverse_events(cursor)
        _insert_canonical_tables(cursor)
        _insert_auxiliary_reference_tables(cursor)
        connection.commit()
    finally:
        connection.close()
    return "data/processed/t2d_platform.db"


# Main.
def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite storage artifacts.")
    parser.parse_args()
    print(run())


# CLI entrypoint.
if __name__ == "__main__":
    main()
