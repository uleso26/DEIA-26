# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.router_agent import RouterAgent


pytestmark = pytest.mark.integration


# Verify canonical drug resolution
def test_canonical_drug_resolution(resolver) -> None:
    resolved = resolver.resolve_drug("Wegovy safety")
    assert resolved is not None
    assert resolved["canonical_id"] == "semaglutide"


# Verify router falls back when Ollama returns empty response
def test_router_falls_back_when_ollama_returns_empty_response() -> None:
    router = RouterAgent()
    with patch.object(router.ollama, "enabled", return_value=True), patch.object(
        router.ollama,
        "generate",
        return_value="   ",
    ):
        payload = router.route("What does SURPASS-3 show?")
    assert payload["question_class"] == "Q2"


# Verify empty query defaults to Q0
def test_empty_query_defaults_to_q0() -> None:
    router = RouterAgent()
    with patch.object(router.ollama, "enabled", return_value=False):
        payload = router.route("")
    assert payload["question_class"] == "Q0"


# Verify greeting query routes to Q0 conversation opening
def test_greeting_query_routes_to_q0_conversation_opening(orchestrator) -> None:
    response = orchestrator.run_query("hi")
    assert response["question_class"] == "Q0"
    assert response["metadata"]["route_reason"] == "conversation_opening"
    assert "outside the current t2d enterprise intelligence scope" not in response["answer"].lower()
    assert "i can help with t2d" in response["answer"].lower()


# Verify capability probe routes to Q0 capability response
def test_capability_probe_routes_to_q0_capability_response(orchestrator) -> None:
    response = orchestrator.run_query("what can you do?")
    assert response["question_class"] == "Q0"
    assert response["metadata"]["route_reason"] == "capability_probe"
    assert "first-line treatment" in response["answer"].lower()
    assert "outside the current t2d enterprise intelligence scope" not in response["answer"].lower()


# Verify general disease question routes to Q7
def test_general_disease_question_routes_to_q7(orchestrator) -> None:
    response = orchestrator.run_query("Can diabetes lead to death? How serious is this disease?")
    assert response["question_class"] == "Q7"
    assert "can contribute to death" in response["answer"].lower()


# Verify broad best drug query routes to treatment selection
def test_broad_best_drug_query_routes_to_treatment_selection(orchestrator) -> None:
    response = orchestrator.run_query("which drug is the best effective one for curing the T2D")
    assert response["question_class"] == "Q3"
    assert "no single drug that cures type 2 diabetes" in response["answer"].lower()
    assert "what outcome matters most" in response["answer"].lower()
    assert response["metadata"].get("needs_clarification")


# Verify initial treatment query routes to Q3
def test_initial_treatment_query_routes_to_q3(orchestrator) -> None:
    response = orchestrator.run_query("When a patient is newly diagnosed with T2D, what is the first Rx medicine?")
    assert response["question_class"] == "Q3"
    assert response["metadata"]["route_reason"] == "initial_treatment_selection"
    assert "metformin is the first-line pharmacotherapy" in response["answer"].lower()


# Verify ambiguous T2D scope query requests clarification instead of Q7
def test_ambiguous_t2d_scope_query_requests_clarification_instead_of_q7(orchestrator) -> None:
    response = orchestrator.run_query("Tell me about T2D medicines.")
    assert response["question_class"] == "Q0"
    assert response["metadata"]["route_reason"] == "t2d_scope_clarification"
    assert response["metadata"].get("needs_clarification")
    assert "arbitrary free-form tasks" not in " ".join(response["caveats"]).lower()


# Verify pricing query routes to Q8
def test_pricing_query_routes_to_q8(orchestrator) -> None:
    response = orchestrator.run_query("What is the latest list price difference between semaglutide and tirzepatide?")
    assert response["question_class"] == "Q8"
    assert "does not include a live pricing" in response["answer"].lower()


# Verify personal urgent query routes to Q9
def test_personal_urgent_query_routes_to_q9(orchestrator) -> None:
    response = orchestrator.run_query("I have diabetes and chest pain right now, what should I do?")
    assert response["question_class"] == "Q9"
    assert "not designed for emergency triage" in response["answer"].lower()


# Verify creative request routes to Q0
def test_creative_request_routes_to_q0(orchestrator) -> None:
    response = orchestrator.run_query("Write a poem about tirzepatide.")
    assert response["question_class"] == "Q0"
    assert "outside the current t2d enterprise intelligence scope" in response["answer"].lower()
