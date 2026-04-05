from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(filename: str) -> Any:
    with (FIXTURE_DIR / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


OPENFDA_DATA = _load_fixture("openfda.json")
CLINICAL_TRIALS = _load_fixture("clinical_trials.json")
PUBMED_DOCUMENTS = _load_fixture("pubmed_documents.json")
OPENTARGETS_DATA = _load_fixture("opentargets.json")
CHEMBL_DATA = _load_fixture("chembl.json")
UNIPROT_DATA = _load_fixture("uniprot.json")
EXTERNAL_INTELLIGENCE = _load_fixture("external_intelligence.json")
GUIDELINE_EXCERPTS = _load_fixture("guideline_excerpts.json")
WHO_GHO_DATA = _load_fixture("who_gho.json")
DRUGBANK_OPEN_DATA = _load_fixture("drugbank_open.json")
SYNTHETIC_PATIENT_PROFILES = _load_fixture("synthetic_patient_profiles.json")
GUIDELINE_GRAPH = _load_fixture("guideline_graph.json")


__all__ = [
    "CHEMBL_DATA",
    "CLINICAL_TRIALS",
    "DRUGBANK_OPEN_DATA",
    "EXTERNAL_INTELLIGENCE",
    "GUIDELINE_EXCERPTS",
    "GUIDELINE_GRAPH",
    "OPENFDA_DATA",
    "OPENTARGETS_DATA",
    "PUBMED_DOCUMENTS",
    "SYNTHETIC_PATIENT_PROFILES",
    "UNIPROT_DATA",
    "WHO_GHO_DATA",
]
