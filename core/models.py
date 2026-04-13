# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Define the citation data model shared across the platform
@dataclass
class Citation:
    source: str
    title: str
    reference_id: str
    url: str
    evidence_tier: str
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Define the agent section data model shared across the platform
@dataclass
class AgentSection:
    agent: str
    question_class: str
    summary: str
    citations: list[Citation] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    evidence_tiers: list[str] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        return payload


# Define the query understanding data model shared across the platform
@dataclass
class QueryUnderstanding:
    query: str
    question_class: str
    question_class_name: str
    scope_family: str
    route_reason: str
    interaction_mode: str
    primary_intent: str
    scores: dict[str, int] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    asks_for_comparison: bool = False
    asks_for_best: bool = False
    objective_terms: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_reason: str | None = None
    clarification_prompt: str | None = None
    confidence: str = "medium"
    routing_mode: str = "deterministic"
    ollama_suggested_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Define the final response data model shared across the platform
@dataclass
class FinalResponse:
    question_class: str
    answer: str
    citations: list[Citation]
    caveats: list[str]
    evidence_tiers: list[str]
    trace_id: str
    sections: list[AgentSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_class": self.question_class,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "caveats": self.caveats,
            "evidence_tiers": self.evidence_tiers,
            "trace_id": self.trace_id,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
        }
