# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


# Verify orchestrator pathway response contains disclaimer
def test_orchestrator_pathway_response_contains_disclaimer(orchestrator) -> None:
    response = orchestrator.run_query(
        "For a T2D patient with CKD stage 3, what does NICE guidance suggest after metformin failure?"
    )
    assert response["question_class"] == "Q3"
    assert "not medical advice" in response["answer"].lower()


# Verify trial name query resolves correct trial
def test_trial_name_query_resolves_correct_trial(orchestrator) -> None:
    response = orchestrator.run_query("What does SURPASS-3 show?")
    assert "surpass-3" in response["answer"].lower()
    assert "surpass-2 provides" not in response["answer"].lower()


# Verify target only query lists linked drugs
def test_target_only_query_lists_linked_drugs(orchestrator) -> None:
    response = orchestrator.run_query("Which drugs share the GLP1R mechanism?")
    assert "semaglutide" in response["answer"].lower()
    assert "tirzepatide" in response["answer"].lower()


# Verify guideline selection respects ADA obesity intent
def test_guideline_selection_respects_ada_obesity_intent(orchestrator) -> None:
    response = orchestrator.run_query("ADA pathway after metformin for obesity")
    assert "glp-1 receptor agonist" in response["answer"].lower()
    assert "ada" in response["answer"].lower()


# Verify initial treatment query returns direct metformin answer with guideline citations
def test_initial_treatment_query_returns_direct_metformin_answer_with_guideline_citations(orchestrator) -> None:
    response = orchestrator.run_query("So when a patient is diagnosed with T2D, what is the first Rx medicine?")
    assert response["question_class"] == "Q3"
    assert "metformin is the first-line pharmacotherapy" in response["answer"].lower()
    citation_titles = [item["title"] for item in response["citations"]]
    assert "ADA Standards of Care 2025: initial pharmacotherapy for newly diagnosed type 2 diabetes" in citation_titles
    assert "NICE NG28 2026 update: first-line drug treatment at diagnosis" in citation_titles


# Verify literature update excludes irrelevant GLP-1 review
def test_literature_update_excludes_irrelevant_glp1_review(orchestrator) -> None:
    response = orchestrator.run_query("Give me an evidence update on SGLT2 inhibitors in heart failure")
    assert "oral glp-1 agonists" not in response["answer"].lower()


# Verify pathway response uses clinical context
def test_pathway_response_uses_clinical_context(orchestrator) -> None:
    response = orchestrator.run_query("ADA pathway after metformin for obesity")
    assert response["metadata"].get("clinical_context_used")


# Verify population query uses WHO context
def test_population_query_uses_who_context(orchestrator) -> None:
    response = orchestrator.run_query("What is the diabetes prevalence in the United Kingdom?")
    assert response["question_class"] == "Q6"
    assert "united kingdom" in response["answer"].lower()
    assert response["metadata"].get("population_context_used")


# Verify expanded catalog supports cost sensitive pathway query
def test_expanded_catalog_supports_cost_sensitive_pathway_query(orchestrator) -> None:
    response = orchestrator.run_query("For a cost-sensitive patient after metformin, what does NICE suggest next?")
    assert response["question_class"] == "Q3"
    assert (
        "sulfonylurea" in response["answer"].lower()
        or "pioglitazone" in response["answer"].lower()
    )


# Verify guideline difference query stays in pathway scope
def test_guideline_difference_query_stays_in_pathway_scope(orchestrator) -> None:
    response = orchestrator.run_query(
        "For a patient with obesity after metformin, how does ADA differ from NICE on the next step?"
    )
    assert response["question_class"] == "Q3"
    assert (
        "guideline comparison" in response["answer"].lower()
        or "cross-guideline comparison was requested" in response["answer"].lower()
    )
    assert response["metadata"]["evidence_review"]["status"] in {"limited", "sufficient"}


# Verify trial query exposes execution plan metadata
def test_trial_query_exposes_execution_plan_metadata(orchestrator) -> None:
    response = orchestrator.run_query("What does SURPASS-3 show?")
    assert response["metadata"]["evidence_plan"]["execution_nodes"] == ["trial"]
    assert response["metadata"]["evidence_review"]["status"] == "sufficient"
    assert response["metadata"]["evidence_plan"]["planning_mode"] in {
        "deterministic",
        "react_planner_ollama",
        "deterministic_guarded_fallback",
    }
    assert "allowed_execution_nodes" in response["metadata"]["evidence_plan"]


# Verify trial publication query prefers trial publication over generic recent literature
def test_trial_publication_query_prefers_trial_publication_over_generic_recent_literature(orchestrator) -> None:
    response = orchestrator.run_query(
        "For SURPASS-3, summarize the trial result and the supporting publication evidence."
    )
    citation_titles = [item["title"] for item in response["citations"]]
    assert any("SURPASS-3 publication" == title for title in citation_titles)
    assert "surpass-3" in response["answer"].lower()
