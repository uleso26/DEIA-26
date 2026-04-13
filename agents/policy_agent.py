# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from typing import Any

from core.models import AgentSection
from agents.base_agent import unique_strings


# Define the policy agent that handles clarification and evidence guardrails
class PolicyAgent:
    """Handle clarification and answer-policy responses before domain execution."""

    def clarification(self, understanding: dict[str, Any]) -> AgentSection:
        question_class = understanding["question_class"]
        query = understanding["query"]
        clarification_prompt = understanding.get("clarification_prompt") or (
            "Could you narrow the request a bit so the answer can stay grounded?"
        )

        if question_class == "Q3" and understanding.get("asks_for_best"):
            summary = (
                "There is no single drug that cures Type 2 Diabetes, and there is no universally best option without "
                "a decision goal. " + clarification_prompt
            )
        else:
            summary = clarification_prompt

        return AgentSection(
            agent="Policy Agent",
            question_class=question_class,
            summary=summary,
            citations=[],
            caveats=[
                "The platform asks for clarification when a treatment recommendation would otherwise be underspecified."
            ],
            evidence_tiers=[],
            tool_outputs=[],
            metadata={
                "mode": "clarification",
                "response_policy": "clarification",
                "clarification_needed": True,
                "clarification_reason": understanding.get("clarification_reason"),
                "clarification_prompt": clarification_prompt,
                "original_query": query,
                "suppress_default_q0_caveat": question_class == "Q0",
                "force_deterministic_synthesis": True,
            },
        )

    def insufficient_evidence_metadata(
        self,
        understanding: dict[str, Any],
        evidence_review: dict[str, Any],
    ) -> dict[str, Any]:
        reason = evidence_review.get("reason", "limited_grounding")
        if reason == "partial_guideline_coverage":
            message = (
                "The current evidence bundle only covered part of the requested guideline comparison, so the answer is "
                "being framed as a partial comparison."
            )
        elif reason == "missing_citations":
            message = (
                "The current answer path returned limited source grounding, so it should be treated as a constrained "
                "best-effort response rather than a full evidence synthesis."
            )
        else:
            message = (
                "The current evidence bundle was not strong enough for a precise grounded answer. A narrower query with "
                "a named drug, trial, or guideline branch would produce a better result."
            )
        return {
            "response_policy": "limited_evidence" if evidence_review.get("status") == "limited" else "insufficient_evidence",
            "evidence_review": evidence_review,
            "evidence_note": message,
            "force_deterministic_synthesis": True,
            "caveat_hint": unique_strings([message]),
        }
