from __future__ import annotations

import re
from typing import Any

from agents.prompt_templates import (
    ROUTER_HUMAN_TEMPLATE,
    ROUTER_SYSTEM_TEMPLATE,
    render_ollama_messages,
)
from tools.query_understanding import ENTERPRISE_ROUTE_LABELS, build_query_understanding
from tools.ollama_client import OllamaClient


VALID_ROUTE_LABELS = ENTERPRISE_ROUTE_LABELS


class RouterAgent:
    """Route enterprise-core questions while keeping scope guardrails deterministic."""

    def __init__(self) -> None:
        self.ollama = OllamaClient()

    def understand(self, query: str) -> dict[str, Any]:
        """Return structured routing state for the query."""
        fallback = build_query_understanding(query).to_dict()
        if fallback["question_class"] not in VALID_ROUTE_LABELS:
            return {**fallback, "routing_mode": "deterministic_scoped"}
        if not self.ollama.enabled("router"):
            return {**fallback, "routing_mode": "deterministic"}

        system, prompt = render_ollama_messages(
            ROUTER_SYSTEM_TEMPLATE,
            ROUTER_HUMAN_TEMPLATE,
            query=query,
        )
        response = self.ollama.generate(
            prompt=prompt,
            system=system,
            model_env="OLLAMA_ROUTER_MODEL",
            default_model="llama3.1:8b",
            timeout_seconds=20,
        )
        match = re.search(r"\bQ[1-6]\b", (response or "").upper())
        label = match.group(0) if match else ""
        # Keep the heuristic route as the guardrail. Small local models are useful here,
        # but they still drift on terse trial acronyms.
        if label in VALID_ROUTE_LABELS and label == fallback["question_class"]:
            return {**fallback, "question_class": label, "routing_mode": "ollama"}
        if label in VALID_ROUTE_LABELS:
            return {
                **fallback,
                "routing_mode": "deterministic_guarded_fallback",
                "ollama_suggested_label": label,
            }
        return {**fallback, "routing_mode": "deterministic_fallback"}

    def route(self, query: str) -> dict[str, Any]:
        """Expose the final routing payload used by the orchestrator."""
        understanding = self.understand(query)
        return {
            "question_class": understanding["question_class"],
            "question_class_name": understanding["question_class_name"],
            "scope_family": understanding["scope_family"],
            "route_reason": understanding["route_reason"],
            "scores": understanding["scores"],
            "routing_mode": understanding.get("routing_mode", "deterministic"),
            "ollama_suggested_label": understanding.get("ollama_suggested_label"),
        }
