from __future__ import annotations

from typing import Any

from agents.base_agent import dedupe_citations, unique_strings
from core.models import AgentSection, FinalResponse
from governance.governance_checker import GovernanceChecker
from tools.ollama_client import OllamaClient


class SynthesisAgent:
    def __init__(self) -> None:
        self.governance = GovernanceChecker()
        self.ollama = OllamaClient()

    def _deterministic_synthesis(self, sections: list[AgentSection]) -> str:
        summaries = [section.summary.strip() for section in sections if section.summary and section.summary.strip()]
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]

        seen_sentences: set[str] = set()
        merged_sentences: list[str] = []
        for summary in summaries:
            sentences = [sentence.strip() for sentence in summary.split(". ") if sentence.strip()]
            for sentence in sentences:
                normalized = sentence.lower().rstrip(".")
                if normalized not in seen_sentences:
                    seen_sentences.add(normalized)
                    merged_sentences.append(sentence.rstrip("."))
        return ". ".join(merged_sentences).strip() + "."

    def run(self, question_class: str, query: str, sections: list[AgentSection], trace_id: str) -> FinalResponse:
        fallback_answer = self._deterministic_synthesis(sections)
        citations = dedupe_citations([citation for section in sections for citation in section.citations])
        caveats = unique_strings([caveat for section in sections for caveat in section.caveats])
        evidence_tiers = unique_strings([tier for section in sections for tier in section.evidence_tiers])
        metadata: dict[str, Any] = {"synthesis_mode": "deterministic"}
        for section in sections:
            metadata.update(section.metadata)
        answer = fallback_answer
        if self.ollama.enabled("synthesis"):
            section_block = "\n\n".join(
                [
                    f"Agent: {section.agent}\nSummary: {section.summary}\nCaveats: {'; '.join(section.caveats)}"
                    for section in sections
                    if section.summary
                ]
            )
            llm_answer = self.ollama.generate(
                prompt=(
                    f"Question class: {question_class}\n"
                    f"User query: {query}\n"
                    f"Evidence summaries:\n{section_block}\n\n"
                    "Write one grounded answer in one or two sentences. Use only the supplied evidence. "
                    "If guideline branches differ, say so explicitly. Do not invent facts."
                ),
                system="You synthesize grounded diabetes evidence into concise clinical intelligence answers.",
                model_env="OLLAMA_SYNTHESIS_MODEL",
                default_model="llama3.1:8b",
                timeout_seconds=45,
            )
            if llm_answer:
                answer = llm_answer
                metadata["synthesis_mode"] = "ollama"
            else:
                metadata["synthesis_mode"] = "deterministic_fallback"
        answer, caveats = self.governance.apply(question_class, answer, caveats, metadata=metadata)
        return FinalResponse(
            question_class=question_class,
            answer=answer,
            citations=citations,
            caveats=caveats,
            evidence_tiers=evidence_tiers,
            trace_id=trace_id,
            sections=sections,
            metadata={"query": query, **metadata},
        )
