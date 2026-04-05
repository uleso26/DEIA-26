from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.router_agent import RouterAgent


pytestmark = pytest.mark.integration


def test_canonical_drug_resolution(resolver) -> None:
    resolved = resolver.resolve_drug("Wegovy safety")
    assert resolved is not None
    assert resolved["canonical_id"] == "semaglutide"


def test_router_falls_back_when_ollama_returns_empty_response() -> None:
    router = RouterAgent()
    with patch.object(router.ollama, "enabled", return_value=True), patch.object(
        router.ollama,
        "generate",
        return_value="   ",
    ):
        payload = router.route("What does SURPASS-3 show?")
    assert payload["question_class"] == "Q2"


def test_empty_query_defaults_to_q0() -> None:
    router = RouterAgent()
    with patch.object(router.ollama, "enabled", return_value=False):
        payload = router.route("")
    assert payload["question_class"] == "Q0"


def test_general_disease_question_routes_to_q7(orchestrator) -> None:
    response = orchestrator.run_query("Can diabetes lead to death? How serious is this disease?")
    assert response["question_class"] == "Q7"
    assert "can contribute to death" in response["answer"].lower()


def test_broad_best_drug_query_routes_to_treatment_selection(orchestrator) -> None:
    response = orchestrator.run_query("which drug is the best effective one for curing the T2D")
    assert response["question_class"] == "Q3"
    assert "no single drug that cures type 2 diabetes" in response["answer"].lower()
    assert "what outcome matters most" in response["answer"].lower()
    assert response["metadata"].get("needs_clarification")


def test_pricing_query_routes_to_q8(orchestrator) -> None:
    response = orchestrator.run_query("What is the latest list price difference between semaglutide and tirzepatide?")
    assert response["question_class"] == "Q8"
    assert "does not include a live pricing" in response["answer"].lower()


def test_personal_urgent_query_routes_to_q9(orchestrator) -> None:
    response = orchestrator.run_query("I have diabetes and chest pain right now, what should I do?")
    assert response["question_class"] == "Q9"
    assert "not designed for emergency triage" in response["answer"].lower()


def test_creative_request_routes_to_q0(orchestrator) -> None:
    response = orchestrator.run_query("Write a poem about tirzepatide.")
    assert response["question_class"] == "Q0"
    assert "outside the current t2d enterprise intelligence scope" in response["answer"].lower()
