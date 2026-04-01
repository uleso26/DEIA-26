from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
