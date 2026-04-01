from __future__ import annotations

import re

from tools.native_tools import ENTERPRISE_ROUTE_LABELS, classify_query
from tools.ollama_client import OllamaClient


VALID_ROUTE_LABELS = ENTERPRISE_ROUTE_LABELS


class RouterAgent:
    def __init__(self) -> None:
        self.ollama = OllamaClient()

    def route(self, query: str) -> dict:
        fallback = classify_query(query)
        if fallback["question_class"] not in VALID_ROUTE_LABELS:
            return {**fallback, "routing_mode": "deterministic_scoped"}
        if not self.ollama.enabled("router"):
            return {**fallback, "routing_mode": "deterministic"}

        response = self.ollama.generate(
            prompt=(
                "Classify the diabetes intelligence request into exactly one label.\n"
                "Q1=safety surveillance\n"
                "Q2=trial efficacy comparison or trial detail\n"
                "Q3=guideline pathway or treatment sequencing\n"
                "Q4=target or mechanism landscape\n"
                "Q5=pipeline or competitor monitoring\n"
                "Q6=literature or population evidence update\n"
                f"Query: {query}\n"
                "Return only the label."
            ),
            system="You are a strict query router. Output one label only: Q1, Q2, Q3, Q4, Q5, or Q6.",
            model_env="OLLAMA_ROUTER_MODEL",
            default_model="llama3.1:8b",
            timeout_seconds=20,
        )
        match = re.search(r"\bQ[1-6]\b", (response or "").upper())
        label = match.group(0) if match else ""
        # Keep the heuristic route as the guardrail. Small local models are useful here,
        # but they still drift on terse trial acronyms.
        if label in VALID_ROUTE_LABELS and label == fallback["question_class"]:
            return {"question_class": label, "scores": fallback["scores"], "routing_mode": "ollama"}
        if label in VALID_ROUTE_LABELS:
            return {
                **fallback,
                "routing_mode": "deterministic_guarded_fallback",
                "ollama_suggested_label": label,
            }
        return {**fallback, "routing_mode": "deterministic_fallback"}
