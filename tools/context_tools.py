"""Context and lookup helpers used by agents and native LangChain tools."""

# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from typing import Any

from core.paths import raw_input_path
from core.storage import connect_sqlite, load_collection, load_json, tokenize
from data.canonical.resolver import get_resolver
from tools.retrieval import search_pubmed


# Define the constants lookup tables and settings used below
COUNTRY_ALIASES = {
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "united states": "United States",
    "usa": "United States",
    "u s a": "United States",
    "america": "United States",
}


# Query the ChEMBL data for matching drug entries
def query_chembl(drug: str) -> list[dict[str, Any]]:
    """Return ChEMBL records for a resolved drug name."""
    resolver = get_resolver()
    resolved = resolver.resolve_drug(drug)
    if not resolved:
        return []
    documents = load_json(raw_input_path("chembl.json"))
    return [item for item in documents if item["canonical_drug"] == resolved["canonical_id"]]


# Query the UniProt data for matching target entries
def query_uniprot(target: str) -> list[dict[str, Any]]:
    """Return UniProt records for a resolved target name."""
    resolver = get_resolver()
    resolved = resolver.resolve_target(target)
    if not resolved:
        return []
    documents = load_json(raw_input_path("uniprot.json"))
    return [item for item in documents if item["canonical_target"] == resolved["canonical_id"]]


# Fetch trial results from the configured source
def fetch_trial_results(query: str) -> list[dict[str, Any]]:
    """Return clinical-trial records matched by trial name or mentioned drugs."""
    resolver = get_resolver()
    matched_trial = resolver.resolve_trial(query)
    documents = load_collection("clinical_trials")
    if matched_trial:
        return [document for document in documents if document["nct_id"] == matched_trial["canonical_id"]]
    matched_drugs = resolver.find_drugs(query)
    if not matched_drugs:
        return documents[:3]
    canonical_drugs = {item["canonical_id"] for item in matched_drugs}
    return [
        document
        for document in documents
        if canonical_drugs.intersection({value.lower() for value in document["interventions"]})
    ]


# Search external intelligence and return the best matches
def search_external_intelligence(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search the external-intelligence collection with simple token overlap."""
    query_tokens = tokenize(query)
    scored = []
    for document in load_collection("external_intelligence"):
        haystack_tokens = set(tokenize(f"{document['headline']} {document['summary']} {' '.join(document['entities_mentioned'])}"))
        score = sum(1 for token in query_tokens if token in haystack_tokens)
        if score:
            item = dict(document)
            item["score"] = score
            scored.append((score, item))
    scored.sort(key=lambda item: (item[0], item[1]["publish_date"]), reverse=True)
    return [item for _, item in scored[:top_k]]


# Fetch population context for the downstream workflow
def get_population_context(country: str | None = None, top_k: int = 3) -> dict[str, Any]:
    """Return WHO population surveillance records from SQLite."""
    connection = connect_sqlite()
    try:
        cursor = connection.cursor()
        select_sql = """
            SELECT country, indicator, year, value, unit, source_url
            FROM who_gho_diabetes_stats
        """
        rows = []
        if country:
            cursor.execute(
                select_sql + " WHERE lower(country) = lower(?) ORDER BY year DESC, value DESC LIMIT ?",
                (country, top_k),
            )
            rows = cursor.fetchall()
            if not rows:
                cursor.execute(
                    select_sql + " WHERE lower(country) LIKE lower(?) ORDER BY year DESC, value DESC LIMIT ?",
                    (f"%{country}%", top_k),
                )
                rows = cursor.fetchall()
        else:
            cursor.execute(select_sql + " ORDER BY year DESC, value DESC LIMIT ?", (top_k,))
            rows = cursor.fetchall()
    finally:
        connection.close()

    records = [
        {
            "country": row[0],
            "indicator": row[1],
            "year": row[2],
            "value": row[3],
            "unit": row[4],
            "source_url": row[5],
        }
        for row in rows
    ]
    return {
        "tool_name": "get_population_context",
        "canonical_entities": {"country": country},
        "source_metadata": {"storage": "sqlite", "table": "who_gho_diabetes_stats"},
        "records": records,
        "raw_provenance": {"country": country, "top_k": top_k},
    }


# Infer country from query from the available query evidence
def infer_country_from_query(query: str) -> str | None:
    """Resolve a supported country alias from a free-form query."""
    token_string = " ".join(tokenize(query))
    for alias, country in COUNTRY_ALIASES.items():
        if alias in token_string:
            return country
    return None


# Fetch clinical context for the downstream workflow
def get_clinical_context(query: str, top_k: int = 2) -> dict[str, Any]:
    """Return DrugBank and synthetic-profile context matched to the query."""
    resolver = get_resolver()
    resolved_drugs = resolver.find_drugs(query)
    canonical_drugs = {item["canonical_id"] for item in resolved_drugs}
    lowered = query.lower()
    class_hints = [
        hint
        for hint in [
            "glp-1 receptor agonist",
            "dual gip/glp-1 receptor agonist",
            "dual gip and glp-1 receptor agonist",
            "sglt2 inhibitor",
            "dpp-4 inhibitor",
            "sulfonylurea",
            "thiazolidinedione",
            "basal insulin",
            "biguanide",
        ]
        if hint in lowered
    ]
    phenotype_terms = [
        term
        for term in [
            "ckd",
            "stage 3",
            "obesity",
            "weight",
            "heart failure",
            "ascvd",
            "metformin",
            "cardiorenal",
            "cost",
            "hypoglycemia",
        ]
        if term in lowered
    ]

    connection = connect_sqlite()
    try:
        connection.row_factory = None
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT canonical_drug, drugbank_id, drug_class, pharmacology, interactions, source_url
            FROM drugbank_classifications
            """
        )
        drugbank_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT patient_id, age_band, sex, bmi_category, ckd_stage, heart_failure,
                   current_therapy, clinical_priority, data_origin
            FROM synthetic_patient_profiles
            """
        )
        synthetic_rows = cursor.fetchall()
    finally:
        connection.close()

    drugbank_matches = []
    for row in drugbank_rows:
        record = {
            "record_type": "drugbank",
            "canonical_drug": row[0],
            "drugbank_id": row[1],
            "drug_class": row[2],
            "pharmacology": row[3],
            "interactions": [item.strip() for item in row[4].split(",") if item.strip()],
            "source_url": row[5],
        }
        if canonical_drugs and record["canonical_drug"] in canonical_drugs:
            drugbank_matches.append(record)
            continue
        drug_class = record["drug_class"].lower()
        if any(hint in drug_class or drug_class in hint for hint in class_hints):
            drugbank_matches.append(record)

    synthetic_matches = []
    for row in synthetic_rows:
        record = {
            "record_type": "synthetic_profile",
            "patient_id": row[0],
            "age_band": row[1],
            "sex": row[2],
            "bmi_category": row[3],
            "ckd_stage": row[4],
            "heart_failure": bool(row[5]),
            "current_therapy": row[6],
            "clinical_priority": row[7],
            "data_origin": row[8],
        }
        score = 0
        if "ckd" in phenotype_terms and record["ckd_stage"] != "none":
            score += 2
        if "stage 3" in phenotype_terms and record["ckd_stage"] == "stage_3":
            score += 2
        if any(term in phenotype_terms for term in ["obesity", "weight"]) and "obesity" in record["bmi_category"]:
            score += 2
        if "heart failure" in phenotype_terms and record["heart_failure"]:
            score += 2
        if "metformin" in phenotype_terms and "metformin" in record["current_therapy"].lower():
            score += 1
        if "cardiorenal" in phenotype_terms and "cardiorenal" in record["clinical_priority"]:
            score += 1
        if "ascvd" in phenotype_terms and "ascvd" in record["clinical_priority"]:
            score += 2
        if "cost" in phenotype_terms and "cost" in record["clinical_priority"]:
            score += 2
        if "hypoglycemia" in phenotype_terms and "hypoglycemia" in record["clinical_priority"]:
            score += 2
        if score:
            record["match_score"] = score
            synthetic_matches.append(record)

    synthetic_matches.sort(key=lambda item: item["match_score"], reverse=True)
    records = [*drugbank_matches[:top_k], *synthetic_matches[:top_k]]
    return {
        "tool_name": "get_clinical_context",
        "canonical_entities": {"drugs": resolved_drugs},
        "source_metadata": {
            "storage": "sqlite",
            "tables": ["drugbank_classifications", "synthetic_patient_profiles"],
        },
        "records": records,
        "raw_provenance": {
            "query": query,
            "class_hints": class_hints,
            "phenotype_terms": phenotype_terms,
        },
    }


# Fetch guideline context for the downstream workflow
def get_guideline_context(query: str) -> list[dict[str, Any]]:
    """Return quick guideline-oriented literature context for a query."""
    return search_pubmed(f"{query} guideline NICE ADA CKD", top_k=2)


__all__ = [
    "COUNTRY_ALIASES",
    "fetch_trial_results",
    "get_clinical_context",
    "get_guideline_context",
    "get_population_context",
    "infer_country_from_query",
    "query_chembl",
    "query_uniprot",
    "search_external_intelligence",
]
