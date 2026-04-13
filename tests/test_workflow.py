# Imports.
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.evidence_planner_agent import EvidencePlannerAgent
from agents.synthesis_agent import SynthesisAgent
from governance.governance_checker import GovernanceChecker
from tools.native_tools import build_evidence_plan, build_query_understanding


pytestmark = pytest.mark.integration


# Test: langgraph workflow exposes expected nodes.
def test_langgraph_workflow_exposes_expected_nodes(orchestrator) -> None:
    mermaid = orchestrator.workflow.mermaid_diagram()
    assert "understand" in mermaid
    assert "policy_gate" in mermaid
    assert "plan" in mermaid
    assert "refine_plan" in mermaid
    assert "execute_plan" in mermaid
    assert "evidence_review" in mermaid
    assert "synthesize" in mermaid
    assert "scope" in mermaid
    assert "clarify" in mermaid


# Test: evidence planner returns bounded trial plan.
def test_evidence_planner_returns_bounded_trial_plan() -> None:
    planner = EvidencePlannerAgent()
    understanding = build_query_understanding("What does SURPASS-3 show?").to_dict()
    base_plan = build_evidence_plan(understanding)
    with patch.object(planner.ollama, "enabled", return_value=False):
        plan = planner.plan(understanding, base_plan)
    assert plan["execution_nodes"] == ["trial"]
    assert plan["planning_mode"] == "deterministic"
    assert plan["react_eligible"]
    assert "literature_q6" in plan["allowed_execution_nodes"]


# Test: evidence planner refines with fallback node when evidence is limited.
def test_evidence_planner_refines_with_fallback_node_when_evidence_is_limited() -> None:
    planner = EvidencePlannerAgent()
    understanding = build_query_understanding("What does SURPASS-3 show?").to_dict()
    base_plan = build_evidence_plan(understanding)
    with patch.object(planner.ollama, "enabled", return_value=False):
        plan = planner.plan(understanding, base_plan)
        refined = planner.refine_after_observation(
            understanding,
            plan,
            sections=[],
            evidence_review={"status": "limited", "reason": "missing_citations"},
            react_steps=0,
            executed_nodes=["trial"],
        )
    assert "literature_q6" in refined["execution_nodes"]
    assert refined["planning_mode"] == "react_refinement_fallback"


# Test: governance checker does not mutate caveats.
def test_governance_checker_does_not_mutate_caveats() -> None:
    caveats = ["Original caveat."]
    original = list(caveats)
    checker = GovernanceChecker()
    _, updated = checker.apply("Q1", "Seed answer", caveats)
    assert caveats == original
    assert len(updated) >= len(original)


# Test: scope routes skip ollama synthesis.
def test_scope_routes_skip_ollama_synthesis() -> None:
    agent = SynthesisAgent()
    with patch.object(agent.ollama, "enabled", return_value=True), patch.object(
        agent.ollama,
        "generate",
    ) as mocked_generate:
        response = agent.run(
            "Q0",
            "hi",
            [],
            "trace-test",
        )
    assert response.metadata["synthesis_mode"] == "deterministic"
    mocked_generate.assert_not_called()
