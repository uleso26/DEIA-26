# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import json
import re
from typing import Any

from agents.prompt_templates import (
    PLAN_REFINEMENT_HUMAN_TEMPLATE,
    PLAN_REFINEMENT_SYSTEM_TEMPLATE,
    PLANNER_HUMAN_TEMPLATE,
    PLANNER_SYSTEM_TEMPLATE,
    render_ollama_messages,
)
from tools.langchain_native_tools import describe_native_tools
from tools.ollama_client import OllamaClient


# Define the constants lookup tables and settings used below
REACT_ELIGIBLE_QUESTION_CLASSES = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
ALLOWED_EXECUTION_NODES = {
    "Q1": ["safety", "literature_q6"],
    "Q2": ["trial", "literature_q6"],
    "Q3": ["pathway", "literature_q6"],
    "Q4": ["knowledge", "molecule", "literature_q6"],
    "Q5": ["literature_q5", "trial", "literature_q6"],
    "Q6": ["literature_q6", "pathway"],
}
EXECUTION_NODE_DESCRIPTIONS = {
    "safety": "Use structured safety surveillance, label, and FAERS-style evidence.",
    "trial": "Use structured clinical trial evidence and head-to-head efficacy context.",
    "pathway": "Use guideline pathway logic and structured clinical decision context.",
    "knowledge": "Use the knowledge graph for mechanism, target, or pathway relationships.",
    "molecule": "Use molecule and pharmacology context for target and mechanism answers.",
    "literature_q5": "Use competitor and pipeline monitoring literature plus external intelligence.",
    "literature_q6": "Use recent literature and population evidence retrieval.",
}
NATIVE_TOOL_HINTS = {
    "Q1": ["search_pubmed_safety_native"],
    "Q2": ["fetch_trial_results_native", "search_retrieval_index_native"],
    "Q3": ["get_clinical_context_native", "search_retrieval_index_native"],
    "Q4": ["get_clinical_context_native", "search_retrieval_index_native"],
    "Q5": ["search_retrieval_index_native"],
    "Q6": ["search_retrieval_index_native", "get_population_context_native"],
}


# Define the evidence planner agent and its specialist response logic
class EvidencePlannerAgent:
    """Bounded evidence planner that can refine execution steps without opening the workflow too far."""

    def __init__(self) -> None:
        self.ollama = OllamaClient()

    @staticmethod
    def _allowed_nodes(question_class: str) -> list[str]:
        return list(ALLOWED_EXECUTION_NODES.get(question_class, []))

    @staticmethod
    def _parse_json_object(payload: str | None) -> dict[str, Any] | None:
        if not payload:
            return None
        text = payload.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _sanitize_node_sequence(candidate_nodes: list[Any], allowed_nodes: list[str], max_steps: int) -> list[str]:
        seen: set[str] = set()
        sanitized: list[str] = []
        for node in candidate_nodes:
            if not isinstance(node, str):
                continue
            normalized = node.strip()
            if normalized not in allowed_nodes or normalized in seen:
                continue
            sanitized.append(normalized)
            seen.add(normalized)
            if len(sanitized) >= max_steps:
                break
        return sanitized

    def plan(self, understanding: dict[str, Any], base_plan: dict[str, Any]) -> dict[str, Any]:
        """Build a bounded execution plan for the current query understanding."""
        question_class = understanding["question_class"]
        if understanding["interaction_mode"] != "enterprise_intelligence":
            return {
                **base_plan,
                "planning_mode": "deterministic_scope",
                "react_eligible": False,
                "max_react_steps": 0,
                "allowed_execution_nodes": base_plan.get("execution_nodes", []),
                "supplemental_nodes": [],
                "native_tool_hints": [],
            }

        allowed_nodes = self._allowed_nodes(question_class)
        default_nodes = self._sanitize_node_sequence(base_plan.get("execution_nodes", []), allowed_nodes, max_steps=3)
        if not default_nodes:
            default_nodes = allowed_nodes[:1]
        native_tool_hints = NATIVE_TOOL_HINTS.get(question_class, [])
        refined_plan = {
            **base_plan,
            "planning_mode": "deterministic",
            "react_eligible": question_class in REACT_ELIGIBLE_QUESTION_CLASSES and len(allowed_nodes) > 1,
            "max_react_steps": 1,
            "allowed_execution_nodes": allowed_nodes,
            "supplemental_nodes": [node for node in allowed_nodes if node not in default_nodes],
            "execution_nodes": default_nodes,
            "native_tool_hints": native_tool_hints,
        }

        if not refined_plan["react_eligible"] or not self.ollama.enabled("planner"):
            return refined_plan

        system, prompt = render_ollama_messages(
            PLANNER_SYSTEM_TEMPLATE,
            PLANNER_HUMAN_TEMPLATE,
            query=understanding["query"],
            question_class=question_class,
            default_nodes=", ".join(default_nodes),
            allowed_nodes=json.dumps(
                [
                    {"name": name, "description": EXECUTION_NODE_DESCRIPTIONS[name]}
                    for name in allowed_nodes
                ],
                ensure_ascii=False,
            ),
            entities=json.dumps(understanding.get("entities", {}), ensure_ascii=False),
            native_tools=json.dumps(describe_native_tools(native_tool_hints), ensure_ascii=False),
        )
        llm_payload = self._parse_json_object(
            self.ollama.generate(
                prompt=prompt,
                system=system,
                model_env="OLLAMA_PLANNER_MODEL",
                default_model="llama3.1:8b",
                timeout_seconds=25,
            )
        )
        candidate_nodes = self._sanitize_node_sequence(
            list(llm_payload.get("execution_nodes", [])) if llm_payload else [],
            allowed_nodes,
            max_steps=3,
        )
        if candidate_nodes:
            refined_plan["execution_nodes"] = candidate_nodes
            refined_plan["supplemental_nodes"] = [node for node in allowed_nodes if node not in candidate_nodes]
            refined_plan["planning_mode"] = "react_planner_ollama"
            if isinstance(llm_payload.get("reason"), str) and llm_payload["reason"].strip():
                refined_plan["planner_reason"] = llm_payload["reason"].strip()
        else:
            refined_plan["planning_mode"] = "deterministic_guarded_fallback"
        return refined_plan

    def refine_after_observation(
        self,
        understanding: dict[str, Any],
        evidence_plan: dict[str, Any],
        sections: list[Any],
        evidence_review: dict[str, Any],
        *,
        react_steps: int,
        executed_nodes: list[str],
    ) -> dict[str, Any]:
        """Optionally add one more allowed node after a limited evidence review."""
        updated = dict(evidence_plan)
        if (
            understanding["interaction_mode"] != "enterprise_intelligence"
            or evidence_review.get("status") not in {"limited", "insufficient"}
            or not updated.get("react_eligible")
            or react_steps >= int(updated.get("max_react_steps", 0))
        ):
            return updated

        available_nodes = [
            node
            for node in updated.get("supplemental_nodes", [])
            if node not in executed_nodes
        ]
        if not available_nodes:
            return updated

        chosen_nodes: list[str] = []
        if self.ollama.enabled("planner"):
            section_summaries = [getattr(section, "summary", "") for section in sections if getattr(section, "summary", "")]
            system, prompt = render_ollama_messages(
                PLAN_REFINEMENT_SYSTEM_TEMPLATE,
                PLAN_REFINEMENT_HUMAN_TEMPLATE,
                query=understanding["query"],
                evidence_review=json.dumps(evidence_review, ensure_ascii=False),
                current_summaries="\n".join(section_summaries),
                available_nodes=json.dumps(
                    [
                        {"name": name, "description": EXECUTION_NODE_DESCRIPTIONS[name]}
                        for name in available_nodes
                    ],
                    ensure_ascii=False,
                ),
            )
            llm_payload = self._parse_json_object(
                self.ollama.generate(
                    prompt=prompt,
                    system=system,
                    model_env="OLLAMA_PLANNER_MODEL",
                    default_model="llama3.1:8b",
                    timeout_seconds=25,
                )
            )
            chosen_nodes = self._sanitize_node_sequence(
                list(llm_payload.get("add_nodes", [])) if llm_payload else [],
                available_nodes,
                max_steps=1,
            )
            if chosen_nodes:
                updated["planning_mode"] = "react_refinement_ollama"
                if isinstance(llm_payload.get("reason"), str) and llm_payload["reason"].strip():
                    updated["refinement_reason"] = llm_payload["reason"].strip()

        if not chosen_nodes:
            fallback_node = self._deterministic_fallback_node(understanding["question_class"], available_nodes, evidence_review)
            if fallback_node:
                chosen_nodes = [fallback_node]
                updated["planning_mode"] = "react_refinement_fallback"

        if not chosen_nodes:
            return updated

        updated["execution_nodes"] = [*updated.get("execution_nodes", []), *chosen_nodes]
        updated["supplemental_nodes"] = [node for node in updated.get("supplemental_nodes", []) if node not in chosen_nodes]
        return updated

    @staticmethod
    def _deterministic_fallback_node(
        question_class: str,
        available_nodes: list[str],
        evidence_review: dict[str, Any],
    ) -> str | None:
        if question_class in {"Q1", "Q2", "Q3", "Q4", "Q5"} and "literature_q6" in available_nodes:
            return "literature_q6"
        if question_class == "Q4" and "molecule" in available_nodes:
            return "molecule"
        if question_class == "Q5" and "trial" in available_nodes:
            return "trial"
        if question_class == "Q6" and "pathway" in available_nodes and evidence_review.get("reason") == "low_routing_confidence":
            return "pathway"
        return available_nodes[0] if available_nodes else None
