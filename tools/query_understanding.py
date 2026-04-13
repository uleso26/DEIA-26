"""Query classification, structured understanding, and evidence-plan helpers."""

# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from typing import Any

from core.models import QueryUnderstanding
from data.canonical.resolver import get_resolver
from tools.context_tools import infer_country_from_query


# Define the constants lookup tables and settings used below
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
GREETING_TERMS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "hiya",
}
CAPABILITY_PROBE_TERMS = {
    "what can you do",
    "how can you help",
    "help",
    "what do you do",
    "what can this do",
}
INITIAL_TREATMENT_TERMS = {
    "first rx",
    "first-line",
    "first line",
    "initial therapy",
    "initial treatment",
    "newly diagnosed",
    "just diagnosed",
    "starting medicine",
    "starting medication",
    "first prescription",
    "start treatment",
    "start with",
    "first medicine",
    "first medication",
}
INITIAL_TREATMENT_CONTEXT_TERMS = {
    "medicine",
    "medication",
    "drug",
    "therapy",
    "treatment",
    "rx",
    "prescription",
    "pharmacotherapy",
}
DISEASE_BACKGROUND_TERMS = {
    "what is t2d",
    "what is type 2 diabetes",
    "what is diabetes",
    "type 2 diabetes is",
    "complication",
    "complications",
    "serious",
    "dangerous",
    "fatal",
    "life expectancy",
    "lead to death",
    "die from diabetes",
    "affect kidneys",
    "affect eyes",
    "affect heart",
}


# Return the user facing name for a routed question class
def question_class_name(question_class: str) -> str:
    """Return the user-facing label for a routed question class."""
    return QUESTION_CLASS_DETAILS.get(question_class, {}).get("name", question_class)


# Return the interaction mode that matches the routed question
def interaction_mode_name(question_class: str, route_reason: str | None = None) -> str:
    """Map a routed question to the broader interaction mode."""
    if route_reason == "conversation_opening":
        return "conversation_opening"
    if route_reason == "capability_probe":
        return "capability_probe"
    if route_reason == "t2d_scope_clarification":
        return "needs_clarification"
    scope_family = QUESTION_CLASS_DETAILS.get(question_class, {}).get("scope_family", "scope_guardrail")
    return INTERACTION_MODE_BY_SCOPE.get(scope_family, "out_of_scope")


# Return the primary intent label that guides downstream synthesis
def primary_intent_name(question_class: str, route_reason: str | None = None) -> str:
    """Map a routed question to the more specific response intent."""
    if route_reason == "conversation_opening":
        return "conversation_opening"
    if route_reason == "capability_probe":
        return "capability_probe"
    if route_reason == "initial_treatment_selection":
        return "initial_treatment"
    if route_reason == "t2d_scope_clarification":
        return "scope_clarification"
    return PRIMARY_INTENT_BY_QUESTION_CLASS[question_class]


# Create an empty score map for question class ranking
def _empty_scores() -> dict[str, int]:
    return {question_class: 0 for question_class in QUESTION_CLASS_DETAILS}


# Build the shared routing payload returned to the workflow
def _route_payload(question_class: str, scores: dict[str, int], route_reason: str) -> dict[str, Any]:
    details = QUESTION_CLASS_DETAILS[question_class]
    return {
        "question_class": question_class,
        "question_class_name": details["name"],
        "scope_family": details["scope_family"],
        "route_reason": route_reason,
        "scores": scores,
    }


# Infer objective terms from the available query evidence
def _infer_objective_terms(lowered: str) -> list[str]:
    objectives: list[str] = []
    for objective_name, hints in DECISION_OBJECTIVE_HINTS.items():
        if any(term in lowered for term in hints):
            objectives.append(objective_name)
    return objectives


# Check whether capability probe applies to the current input
def _is_capability_probe(lowered: str) -> bool:
    return lowered in CAPABILITY_PROBE_TERMS or any(term in lowered for term in CAPABILITY_PROBE_TERMS if len(term) > 4)


# Check whether initial treatment query applies to the current input
def _is_initial_treatment_query(lowered: str) -> bool:
    has_initial_marker = any(term in lowered for term in INITIAL_TREATMENT_TERMS)
    if not has_initial_marker:
        return False
    return any(term in lowered for term in INITIAL_TREATMENT_CONTEXT_TERMS) or any(
        term in lowered
        for term in [
            "diagnosed with t2d",
            "diagnosed with type 2 diabetes",
            "type 2 diabetes",
            "t2d",
        ]
    )


# Check whether explicit disease background query applies to the current input
def _is_explicit_disease_background_query(lowered: str) -> bool:
    return any(term in lowered for term in DISEASE_BACKGROUND_TERMS)


# Convert question class scores into a simple routing confidence value
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


# Extract query entities from the upstream payload
def extract_query_entities(query: str) -> dict[str, Any]:
    """Resolve the main structured entities needed for routing and planning."""
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


# Build query understanding for the downstream execution path
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
    if route["question_class"] == "Q3" and asks_for_best and not objective_terms and route["route_reason"] != "initial_treatment_selection":
        needs_clarification = True
        clarification_reason = "treatment_goal_missing"
        clarification_prompt = (
            "What outcome matters most here: HbA1c lowering, weight loss, CKD/HF benefit, "
            "lower hypoglycaemia risk, or lower cost?"
        )
    elif route["route_reason"] == "t2d_scope_clarification":
        needs_clarification = True
        clarification_reason = "t2d_intent_unspecified"
        clarification_prompt = (
            "Please narrow this to one T2D task such as first-line treatment, a guideline pathway, "
            "a trial result, a safety signal, a mechanism question, or a literature summary."
        )

    return QueryUnderstanding(
        query=query,
        question_class=route["question_class"],
        question_class_name=route["question_class_name"],
        scope_family=route["scope_family"],
        route_reason=route["route_reason"],
        interaction_mode=interaction_mode_name(route["question_class"], route["route_reason"]),
        primary_intent=primary_intent_name(route["question_class"], route["route_reason"]),
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


# Build evidence plan for the downstream execution path
def build_evidence_plan(understanding: dict[str, Any]) -> dict[str, Any]:
    """Build the first-pass execution plan before bounded refinement."""
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
        "plan_name": understanding.get("primary_intent", PRIMARY_INTENT_BY_QUESTION_CLASS[question_class]),
        "execution_nodes": execution_nodes,
        "requires_evidence_review": True,
    }


# Assess whether the collected sections are sufficient for synthesis
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


# Check whether domain context is present before branching
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


# Classify the raw query into the platform question taxonomy
def classify_query(query: str) -> dict[str, Any]:
    """Classify a query into the platform scope lanes using deterministic heuristics."""
    lowered = query.lower().strip()
    normalized_lowered = lowered.strip("!?., ")
    scores = _empty_scores()
    if not lowered:
        return _route_payload("Q0", scores, "empty_query")
    if normalized_lowered in GREETING_TERMS:
        return _route_payload("Q0", scores, "conversation_opening")
    if normalized_lowered in CAPABILITY_PROBE_TERMS or _is_capability_probe(lowered):
        return _route_payload("Q0", scores, "capability_probe")

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
    if domain_context and _is_initial_treatment_query(lowered):
        scores["Q3"] = 3
        return _route_payload("Q3", scores, "initial_treatment_selection")
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
        if domain_context and _is_explicit_disease_background_query(lowered):
            scores["Q7"] = 1
            return _route_payload("Q7", scores, "general_disease_background")
        if domain_context:
            return _route_payload("Q0", scores, "t2d_scope_clarification")
        return _route_payload("Q0", scores, "outside_t2d_scope")

    tied = [question_class for question_class in ENTERPRISE_ROUTE_LABELS if scores[question_class] == top_score]
    question_class = next(
        question_class
        for question_class in QUESTION_TIE_BREAK_ORDER
        if question_class in tied
    )
    return _route_payload(question_class, scores, "enterprise_core")


__all__ = [
    "ENTERPRISE_ROUTE_LABELS",
    "PRIMARY_INTENT_BY_QUESTION_CLASS",
    "QUESTION_CLASS_DETAILS",
    "assess_evidence_sufficiency",
    "build_evidence_plan",
    "build_query_understanding",
    "classify_query",
    "extract_query_entities",
    "interaction_mode_name",
    "question_class_name",
]
