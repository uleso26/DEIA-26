"""Query understanding, retrieval helpers, and local evidence-tool utilities."""

from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Any

from core.models import QueryUnderstanding
from core.paths import raw_input_path
from core.storage import (
    build_chroma_index,
    build_lexical_index,
    chunk_retrieval_documents,
    connect_sqlite,
    load_collection,
    load_json,
    load_retrieval_manifest,
    search_dense_index,
    search_lexical_index,
    tokenize,
)
from data.canonical.resolver import get_resolver


# Hybrid retrieval leans slightly toward dense search because the corpus is
# small and terminology can vary, but lexical overlap still helps anchor exact
# trial names and guideline phrasing.
LEXICAL_FUSION_WEIGHT = 0.4
DENSE_FUSION_WEIGHT = 0.6

# Domain reranking heuristics keep obvious entity and intent hits at the top
# after lexical+dense fusion has narrowed the field.
DRUG_MATCH_BOOST = 6.0
TARGET_MATCH_BOOST = 4.0
SAFETY_INTENT_BOOST = 3.0
POST_MARKETING_BOOST = 2.0
SURVEILLANCE_BOOST = 1.5
HEART_FAILURE_BOOST = 3.0


QUESTION_CLASS_DETAILS = {
    "Q0": {
        "name": "Out-of-Scope Or Clarification",
        "scope_family": "scope_guardrail",
    },
    "Q1": {
        "name": "Safety Surveillance",
        "scope_family": "enterprise_core",
    },
    "Q2": {
        "name": "Trial And Efficacy Intelligence",
        "scope_family": "enterprise_core",
    },
    "Q3": {
        "name": "Guideline And Sequencing Intelligence",
        "scope_family": "enterprise_core",
    },
    "Q4": {
        "name": "Mechanism And Target Intelligence",
        "scope_family": "enterprise_core",
    },
    "Q5": {
        "name": "Competitive And Pipeline Intelligence",
        "scope_family": "enterprise_core",
    },
    "Q6": {
        "name": "Literature And Population Evidence",
        "scope_family": "enterprise_core",
    },
    "Q7": {
        "name": "Disease Background And Risk Communication",
        "scope_family": "medical_background",
    },
    "Q8": {
        "name": "Pricing, Access, And Market Access Scope",
        "scope_family": "commercial_scope",
    },
    "Q9": {
        "name": "Urgent Or Personal Medical Guardrail",
        "scope_family": "urgent_guardrail",
    },
}

QUESTION_KEYWORDS = {
    "Q1": ["adverse", "side effect", "safety", "faers", "signal", "post-marketing", "warning", "label", "tolerability"],
    "Q2": ["compare", "trial", "phase", "hba1c", "efficacy", "head-to-head", "versus", "vs", "endpoint", "weight loss", "outcomes"],
    "Q3": [
        "guidance",
        "guideline",
        "after metformin",
        "next step",
        "ckd",
        "pathway",
        "ada",
        "nice",
        "sequencing",
        "treatment pathway",
        "ascvd",
        "heart failure",
        "hypoglycemia",
        "cost",
        "insulin start",
        "fatty liver",
        "masld",
        "nash",
    ],
    "Q4": ["target", "mechanism", "acts on", "share this mechanism", "protein", "receptor", "co-agonist", "landscape"],
    "Q5": ["latest", "pipeline", "competitor", "monitoring", "developments", "oral glp-1", "readout", "external intelligence"],
    "Q6": [
        "summarize publications",
        "last 6 months",
        "evidence update",
        "recent literature",
        "journal",
        "prevalence",
        "burden",
        "epidemiology",
        "population",
        "meta-analysis",
        "real-world",
        "mortality",
        "survival",
        "incidence",
    ],
}

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

QUESTION_TIE_BREAK_ORDER = ["Q2", "Q3", "Q1", "Q4", "Q5", "Q6"]
ENTERPRISE_ROUTE_LABELS = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
INTERACTION_MODE_BY_SCOPE = {
    "enterprise_core": "enterprise_intelligence",
    "medical_background": "general_background",
    "commercial_scope": "commercial_scope",
    "urgent_guardrail": "urgent_guardrail",
    "scope_guardrail": "out_of_scope",
}
PRIMARY_INTENT_BY_QUESTION_CLASS = {
    "Q0": "out_of_scope",
    "Q1": "safety_signal",
    "Q2": "trial_or_efficacy_intelligence",
    "Q3": "treatment_selection_or_guideline_pathway",
    "Q4": "mechanism_or_target_landscape",
    "Q5": "competitor_or_pipeline_monitoring",
    "Q6": "literature_or_population_evidence",
    "Q7": "disease_background",
    "Q8": "pricing_and_market_access_scope",
    "Q9": "personal_or_urgent_medical",
}
DOMAIN_HINT_TERMS = {
    "diabetes",
    "type 2 diabetes",
    "t2d",
    "glucose",
    "blood sugar",
    "hyperglycemia",
    "hypoglycemia",
    "metformin",
    "insulin",
    "semaglutide",
    "tirzepatide",
    "empagliflozin",
    "dapagliflozin",
    "canagliflozin",
    "ertugliflozin",
    "liraglutide",
    "dulaglutide",
    "exenatide",
    "lixisenatide",
    "sitagliptin",
    "linagliptin",
    "saxagliptin",
    "alogliptin",
    "pioglitazone",
    "gliclazide",
    "glimepiride",
    "glipizide",
    "insulin glargine",
    "insulin degludec",
    "repaglinide",
    "acarbose",
    "orforglipron",
}
PRICING_ACCESS_TERMS = {
    "price",
    "pricing",
    "cost",
    "reimbursement",
    "payer",
    "formulary",
    "access",
    "coverage",
    "co-pay",
    "copay",
    "market access",
    "list price",
    "net price",
    "wac",
    "contract",
}
TREATMENT_SELECTION_TERMS = {
    "best drug",
    "best medicine",
    "best medication",
    "most effective",
    "best effective",
    "strongest",
    "curing",
    "cure",
}
DECISION_OBJECTIVE_HINTS = {
    "hba1c_lowering": {"hba1c", "glycaemic", "glycemic", "glucose", "a1c"},
    "weight_loss": {"weight", "obesity", "bmi"},
    "cardiorenal_benefit": {"ckd", "kidney", "renal", "heart failure", "hf", "ascvd", "cardiovascular"},
    "low_hypoglycemia_risk": {"hypoglycemia", "hypoglycaemia", "avoid lows"},
    "lower_cost": {"cost", "cheap", "afford", "price", "pricing"},
}
PERSONAL_URGENT_TERMS = {
    "i have",
    "my ",
    "should i",
    "what should i do",
    "do i need",
    "am i",
    "for me",
    "right now",
    "help me",
}
URGENT_MEDICAL_TERMS = {
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "trouble breathing",
    "passed out",
    "fainted",
    "confusion",
    "severe vomiting",
    "ketones",
    "ketoacidosis",
    "dka",
    "hospital now",
    "emergency",
}
NON_ENTERPRISE_REQUEST_TERMS = {
    "poem",
    "joke",
    "story",
    "song",
    "rap",
    "haiku",
    "email draft",
    "cover letter",
}


def question_class_name(question_class: str) -> str:
    return QUESTION_CLASS_DETAILS.get(question_class, {}).get("name", question_class)


def interaction_mode_name(question_class: str) -> str:
    scope_family = QUESTION_CLASS_DETAILS.get(question_class, {}).get("scope_family", "scope_guardrail")
    return INTERACTION_MODE_BY_SCOPE.get(scope_family, "out_of_scope")


def _empty_scores() -> dict[str, int]:
    return {question_class: 0 for question_class in QUESTION_CLASS_DETAILS}


def _route_payload(question_class: str, scores: dict[str, int], route_reason: str) -> dict[str, Any]:
    details = QUESTION_CLASS_DETAILS[question_class]
    return {
        "question_class": question_class,
        "question_class_name": details["name"],
        "scope_family": details["scope_family"],
        "route_reason": route_reason,
        "scores": scores,
    }


def _infer_objective_terms(lowered: str) -> list[str]:
    objectives: list[str] = []
    for objective_name, hints in DECISION_OBJECTIVE_HINTS.items():
        if any(term in lowered for term in hints):
            objectives.append(objective_name)
    return objectives


def _confidence_from_scores(question_class: str, scores: dict[str, int]) -> str:
    if question_class not in ENTERPRISE_ROUTE_LABELS:
        return "high"
    ranked = sorted((scores.get(label, 0) for label in ENTERPRISE_ROUTE_LABELS), reverse=True)
    top_score = ranked[0] if ranked else 0
    second_score = ranked[1] if len(ranked) > 1 else 0
    if top_score <= 1:
        return "low"
    if top_score - second_score >= 2:
        return "high"
    return "medium"


def extract_query_entities(query: str) -> dict[str, Any]:
    resolver = get_resolver()
    resolved_trial = resolver.resolve_trial(query)
    resolved_target = resolver.resolve_target(query)
    resolved_drugs = resolver.find_drugs(query)
    country = infer_country_from_query(query)
    return {
        "trial_id": (resolved_trial or {}).get("canonical_id"),
        "target_id": (resolved_target or {}).get("canonical_id"),
        "drug_ids": [item["canonical_id"] for item in resolved_drugs],
        "country": country,
    }


def build_query_understanding(query: str) -> QueryUnderstanding:
    """Turn a raw query into the structured routing state used by the workflow."""
    route = classify_query(query)
    lowered = query.lower().strip()
    entities = extract_query_entities(query)
    objective_terms = _infer_objective_terms(lowered)
    asks_for_comparison = any(token in lowered for token in ["compare", "comparison", "difference", "differ", "versus", " vs "])
    asks_for_best = any(term in lowered for term in TREATMENT_SELECTION_TERMS) or (
        any(term in lowered for term in ["which drug", "which medicine", "which medication"])
        and any(term in lowered for term in ["best", "effective", "strongest"])
    )

    needs_clarification = False
    clarification_reason = None
    clarification_prompt = None
    if route["question_class"] == "Q3" and asks_for_best and not objective_terms:
        needs_clarification = True
        clarification_reason = "treatment_goal_missing"
        clarification_prompt = (
            "What outcome matters most here: HbA1c lowering, weight loss, CKD/HF benefit, "
            "lower hypoglycaemia risk, or lower cost?"
        )

    return QueryUnderstanding(
        query=query,
        question_class=route["question_class"],
        question_class_name=route["question_class_name"],
        scope_family=route["scope_family"],
        route_reason=route["route_reason"],
        interaction_mode=interaction_mode_name(route["question_class"]),
        primary_intent=PRIMARY_INTENT_BY_QUESTION_CLASS[route["question_class"]],
        scores=route["scores"],
        entities=entities,
        asks_for_comparison=asks_for_comparison,
        asks_for_best=asks_for_best,
        objective_terms=objective_terms,
        needs_clarification=needs_clarification,
        clarification_reason=clarification_reason,
        clarification_prompt=clarification_prompt,
        confidence=_confidence_from_scores(route["question_class"], route["scores"]),
    )


def build_evidence_plan(understanding: dict[str, Any]) -> dict[str, Any]:
    question_class = understanding["question_class"]
    if understanding["interaction_mode"] != "enterprise_intelligence":
        return {
            "plan_name": "direct_scope_response",
            "execution_nodes": ["scope"],
            "requires_evidence_review": False,
        }
    lowered_query = understanding["query"].lower()
    execution_nodes = {
        "Q1": ["safety"],
        "Q2": ["trial"],
        "Q3": ["pathway"],
        "Q4": ["knowledge", "molecule"],
        "Q5": ["literature_q5"],
        "Q6": ["literature_q6"],
    }[question_class]
    if question_class in {"Q2", "Q3"} and any(term in lowered_query for term in ["publication", "recent", "evidence", "literature", "journal"]):
        execution_nodes = [*execution_nodes, "literature_q6"]
    if question_class == "Q1" and any(term in lowered_query for term in ["publication", "recent", "evidence"]):
        execution_nodes = [*execution_nodes, "literature_q6"]
    return {
        "plan_name": PRIMARY_INTENT_BY_QUESTION_CLASS[question_class],
        "execution_nodes": execution_nodes,
        "requires_evidence_review": True,
    }


def assess_evidence_sufficiency(
    understanding: dict[str, Any],
    sections: list[dict[str, Any] | Any],
) -> dict[str, Any]:
    """Judge whether the gathered sections are grounded enough to synthesize."""
    if understanding["interaction_mode"] != "enterprise_intelligence":
        return {"status": "not_applicable", "reason": "non_enterprise_scope"}

    normalized_sections = []
    for section in sections:
        if hasattr(section, "to_dict"):
            normalized_sections.append(section.to_dict())
        else:
            normalized_sections.append(dict(section))

    if not normalized_sections:
        return {"status": "insufficient", "reason": "no_sections"}

    citations_count = sum(len(section.get("citations", [])) for section in normalized_sections)
    summaries = [str(section.get("summary", "")).strip() for section in normalized_sections]
    combined_summary = " ".join(summary for summary in summaries if summary).lower()
    if not combined_summary:
        return {"status": "insufficient", "reason": "empty_summary"}
    if any(
        phrase in combined_summary
        for phrase in [
            "no trial comparison data was found",
            "no pathway logic was found",
            "no graph targets were found",
            "no drug or target entities were resolved",
            "no recent literature was found",
            "no competitor monitoring evidence was found",
        ]
    ):
        return {"status": "insufficient", "reason": "no_grounded_records"}
    if citations_count == 0:
        return {"status": "limited", "reason": "missing_citations"}
    if "one guideline only" in combined_summary:
        return {"status": "limited", "reason": "partial_guideline_coverage"}
    if understanding.get("confidence") == "low":
        return {"status": "limited", "reason": "low_routing_confidence"}
    return {"status": "sufficient", "reason": "grounded_evidence_present"}


def _has_domain_context(query: str, lowered: str) -> bool:
    resolver = get_resolver()
    if any(term in lowered for term in DOMAIN_HINT_TERMS):
        return True
    if resolver.resolve_trial(query):
        return True
    if resolver.resolve_target(query):
        return True
    if resolver.find_drugs(query):
        return True
    return False


def classify_query(query: str) -> dict[str, Any]:
    """Classify a query into the platform scope lanes using deterministic heuristics."""
    lowered = query.lower().strip()
    scores = _empty_scores()
    if not lowered:
        return _route_payload("Q0", scores, "empty_query")

    domain_context = _has_domain_context(query, lowered)
    guideline_intent = any(
        term in lowered
        for term in ["ada", "nice", "guideline", "pathway", "after metformin", "next step", "sequencing"]
    )
    if any(term in lowered for term in NON_ENTERPRISE_REQUEST_TERMS):
        return _route_payload("Q0", scores, "non_intelligence_request")
    if any(term in lowered for term in URGENT_MEDICAL_TERMS) or (
        domain_context and any(term in lowered for term in PERSONAL_URGENT_TERMS)
    ):
        scores["Q9"] = 3
        return _route_payload("Q9", scores, "personal_or_urgent_medical")
    if domain_context and not guideline_intent and any(term in lowered for term in PRICING_ACCESS_TERMS):
        scores["Q8"] = 3
        return _route_payload("Q8", scores, "pricing_or_market_access")
    if domain_context and (
        any(term in lowered for term in TREATMENT_SELECTION_TERMS)
        or (
            any(term in lowered for term in ["which drug", "which medicine", "which medication"])
            and any(term in lowered for term in ["best", "effective", "cure", "curing", "treatment", "therapy"])
        )
    ):
        scores["Q3"] = 3
        return _route_payload("Q3", scores, "treatment_selection")

    resolver = get_resolver()
    for question_class, keywords in QUESTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[question_class] += 1

    if "last 6 months" in lowered or "publications" in lowered:
        scores["Q6"] += 2
    if "literature" in lowered or "journal" in lowered:
        scores["Q6"] += 2
    if "latest" in lowered or "pipeline" in lowered:
        scores["Q5"] += 2
    if "ckd" in lowered and ("after metformin" in lowered or "guid" in lowered):
        scores["Q3"] += 2
    if "compare" in lowered and ("semaglutide" in lowered or "tirzepatide" in lowered):
        scores["Q2"] += 2
    if any(term in lowered for term in ["surpass", "sustain forte", "dapa-hf", "leader", "empa-reg", "empareg"]):
        scores["Q2"] += 3
    if "oral glp-1" in lowered and ("literature" in lowered or "publications" in lowered):
        scores["Q6"] += 2
        scores["Q5"] = max(0, scores["Q5"] - 1)
    if any(term in lowered for term in ["prevalence", "burden", "epidemiology", "population"]):
        scores["Q6"] += 2
    if any(term in lowered for term in ["vs", "versus", "superiority", "noninferiority"]):
        scores["Q2"] += 1
    if resolver.resolve_trial(query):
        scores["Q2"] += 3
    matched_drugs = resolver.find_drugs(query)
    if len(matched_drugs) >= 2 and any(term in lowered for term in ["vs", "versus", "compare", "difference"]):
        scores["Q2"] += 2
    if resolver.resolve_target(query):
        scores["Q4"] += 2
    if any(term in lowered for term in ["mechanism", "target", "receptor", "co-agonist", "protein"]):
        scores["Q4"] += 2
    if guideline_intent:
        scores["Q3"] += 1

    top_score = max(scores[question_class] for question_class in ENTERPRISE_ROUTE_LABELS)
    if top_score == 0:
        if domain_context:
            scores["Q7"] = 1
            return _route_payload("Q7", scores, "general_disease_background")
        return _route_payload("Q0", scores, "outside_t2d_scope")
    else:
        tied = [question_class for question_class in ENTERPRISE_ROUTE_LABELS if scores[question_class] == top_score]
        question_class = next(
            question_class
            for question_class in QUESTION_TIE_BREAK_ORDER
            if question_class in tied
        )
    return _route_payload(question_class, scores, "enterprise_core")


def _pubmed_documents() -> list[dict[str, Any]]:
    return load_json(raw_input_path("pubmed_documents.json"))


def _retrieval_documents() -> list[dict[str, Any]]:
    documents = []
    source_documents = [*_pubmed_documents()]
    guideline_path = raw_input_path("guideline_excerpts.json")
    if guideline_path.exists():
        source_documents.extend(load_json(guideline_path))
    for document in source_documents:
        enriched = dict(document)
        enriched["retrieval_text"] = " ".join(
            [
                str(document.get("title", "")),
                str(document.get("journal", "")),
                " ".join(document.get("mesh_terms", [])),
                str(document.get("text", "")),
            ]
        ).strip()
        documents.append(enriched)
    return documents


def search_retrieval_index(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Run hybrid retrieval over chunked literature and guideline evidence."""
    manifest = load_retrieval_manifest()
    if not manifest.get("documents"):
        fallback_documents = chunk_retrieval_documents(_retrieval_documents(), text_key="retrieval_text")
        manifest = build_lexical_index(fallback_documents, text_key="chunk_text")
        manifest["documents"] = fallback_documents
        manifest["backend"] = "lexical"
        chroma_manifest = build_chroma_index(fallback_documents, text_key="chunk_text")
        if chroma_manifest:
            manifest["chroma_collection"] = chroma_manifest["collection_name"]
            manifest["embedding_provider"] = chroma_manifest["embedding_provider"]
            manifest["embedding_model"] = chroma_manifest["embedding_model"]
            manifest["vector_dim"] = chroma_manifest["vector_dim"]
            manifest["backend"] = "hybrid_chroma"
        else:
            manifest["dense_vectors"] = {}
    candidate_count = max(top_k * 4, top_k)
    lexical_results = search_lexical_index(query, manifest, top_k=candidate_count)
    dense_results = search_dense_index(query, manifest, top_k=candidate_count)
    normalized_scores: dict[str, float] = {}
    for weight, results in [
        (LEXICAL_FUSION_WEIGHT, lexical_results),
        (DENSE_FUSION_WEIGHT, dense_results),
    ]:
        if not results:
            continue
        max_score = max(float(item.get("score", 0.0)) for item in results) or 1.0
        for item in results:
            doc_id = str(item.get("parent_doc_id") or item.get("doc_id") or item.get("pmid") or item.get("id"))
            normalized_scores[doc_id] = normalized_scores.get(doc_id, 0.0) + weight * (float(item.get("score", 0.0)) / max_score)

    initial_results = []
    seen_doc_ids: set[str] = set()
    for item in [*dense_results, *lexical_results]:
        doc_id = str(item.get("parent_doc_id") or item.get("doc_id") or item.get("pmid") or item.get("id"))
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        seeded = dict(item)
        seeded["score"] = round(normalized_scores.get(doc_id, float(item.get("score", 0.0))), 4)
        if seeded.get("chunk_text") and not seeded.get("text"):
            seeded["text"] = seeded["chunk_text"]
        initial_results.append(seeded)

    if not initial_results:
        initial_results = lexical_results
    resolver = get_resolver()
    matched_drugs = {item["canonical_id"] for item in resolver.find_drugs(query)}
    matched_target = (resolver.resolve_target(query) or {}).get("canonical_id")
    lowered_query = query.lower()

    reranked = []
    for item in initial_results:
        # The lexical layer gets us close. This second pass nudges ranking toward
        # drug, target, and intent matches that matter in this domain.
        haystack = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("journal", "")),
                " ".join(item.get("mesh_terms", [])),
                str(item.get("text", "")),
            ]
        ).lower()
        rerank_score = float(item.get("score", 0.0))
        for drug in matched_drugs:
            if drug.lower() in haystack:
                rerank_score += DRUG_MATCH_BOOST
        if matched_target and matched_target.lower() in haystack:
            rerank_score += TARGET_MATCH_BOOST
        if "safety" in lowered_query and any(term in haystack for term in ["safety", "faers", "pharmacovigilance"]):
            rerank_score += SAFETY_INTENT_BOOST
        if "post-marketing" in lowered_query and "post-marketing" in haystack:
            rerank_score += POST_MARKETING_BOOST
        if "surveillance" in lowered_query and any(term in haystack for term in ["surveillance", "pharmacovigilance"]):
            rerank_score += SURVEILLANCE_BOOST
        if "heart failure" in lowered_query and "heart failure" in haystack:
            rerank_score += HEART_FAILURE_BOOST
        item = dict(item)
        item["score"] = round(rerank_score, 4)
        reranked.append(item)

    reranked.sort(key=lambda item: (item.get("score", 0.0), item.get("publication_date", "")), reverse=True)
    return reranked[:top_k]


def search_pubmed(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    documents = _pubmed_documents()
    query_tokens = tokenize(query)
    scored = []
    for document in documents:
        haystack_tokens = set(tokenize(f"{document['title']} {document['text']} {' '.join(document['mesh_terms'])}"))
        score = sum(1 for token in query_tokens if token in haystack_tokens)
        if score:
            item = dict(document)
            item["score"] = score
            scored.append((score, item))
    scored.sort(key=lambda item: (item[0], item[1]["publication_date"]), reverse=True)
    return [item for _, item in scored[:top_k]]


def search_pubmed_safety(drug: str, top_k: int = 2) -> list[dict[str, Any]]:
    return search_pubmed(f"{drug} safety faers adverse", top_k=top_k)


def query_chembl(drug: str) -> list[dict[str, Any]]:
    resolver = get_resolver()
    resolved = resolver.resolve_drug(drug)
    if not resolved:
        return []
    documents = load_json(raw_input_path("chembl.json"))
    return [item for item in documents if item["canonical_drug"] == resolved["canonical_id"]]


def query_uniprot(target: str) -> list[dict[str, Any]]:
    resolver = get_resolver()
    resolved = resolver.resolve_target(target)
    if not resolved:
        return []
    documents = load_json(raw_input_path("uniprot.json"))
    return [item for item in documents if item["canonical_target"] == resolved["canonical_id"]]


def fetch_trial_results(query: str) -> list[dict[str, Any]]:
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


def get_guideline_context(query: str) -> list[dict[str, Any]]:
    return search_pubmed(f"{query} guideline NICE ADA CKD", top_k=2)


def search_external_intelligence(query: str, top_k: int = 3) -> list[dict[str, Any]]:
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


def get_population_context(country: str | None = None, top_k: int = 3) -> dict[str, Any]:
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


def infer_country_from_query(query: str) -> str | None:
    token_string = " ".join(tokenize(query))
    for alias, country in COUNTRY_ALIASES.items():
        if alias in token_string:
            return country
    return None


def get_clinical_context(query: str, top_k: int = 2) -> dict[str, Any]:
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


def filter_recent_documents(documents: list[dict[str, Any]], months: int = 6, reference_date: date | None = None) -> list[dict[str, Any]]:
    """Keep documents within a recent-months window, skipping malformed dates."""
    today = reference_date or date.today()
    threshold_month = today.month - months
    threshold_year = today.year
    while threshold_month <= 0:
        threshold_month += 12
        threshold_year -= 1
    threshold_day = min(today.day, monthrange(threshold_year, threshold_month)[1])
    threshold = date(threshold_year, threshold_month, threshold_day)
    recent = []
    for document in documents:
        try:
            published = date.fromisoformat(str(document["publication_date"]))
        except (TypeError, ValueError):
            continue
        if published >= threshold:
            recent.append(document)
    return recent
