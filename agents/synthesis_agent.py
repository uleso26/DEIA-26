from __future__ import annotations

from typing import Any

from agents.base_agent import dedupe_citations, unique_strings
from agents.prompt_templates import (
    SYNTHESIS_HUMAN_TEMPLATE,
    SYNTHESIS_SYSTEM_TEMPLATE,
    render_ollama_messages,
    section_block_for_prompt,
)
from core.models import AgentSection, FinalResponse
from governance.governance_checker import GovernanceChecker
from tools.ollama_client import OllamaClient


LLM_SYNTHESIS_CLASSES = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}


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

    def run(
        self,
        question_class: str,
        query: str,
        sections: list[AgentSection],
        trace_id: str,
        workflow_metadata: dict[str, Any] | None = None,
    ) -> FinalResponse:
        fallback_answer = self._deterministic_synthesis(sections)
        citations = dedupe_citations([citation for section in sections for citation in section.citations])
        caveats = unique_strings([caveat for section in sections for caveat in section.caveats])
        evidence_tiers = unique_strings([tier for section in sections for tier in section.evidence_tiers])
        metadata: dict[str, Any] = {"synthesis_mode": "deterministic"}
        for section in sections:
            metadata.update(section.metadata)
        if workflow_metadata:
            metadata.update(workflow_metadata)
            caveats = unique_strings([*caveats, *(workflow_metadata.get("caveat_hint") or [])])
        answer = fallback_answer
        should_try_ollama = (
            question_class in LLM_SYNTHESIS_CLASSES
            and self.ollama.enabled("synthesis")
            and any(section.summary for section in sections)
            and not any(section.metadata.get("force_deterministic_synthesis") for section in sections)
            and not metadata.get("force_deterministic_synthesis")
        )
        if should_try_ollama:
            section_block = section_block_for_prompt(sections)
            system, prompt = render_ollama_messages(
                SYNTHESIS_SYSTEM_TEMPLATE,
                SYNTHESIS_HUMAN_TEMPLATE,
                question_class=question_class,
                query=query,
                section_block=section_block,
            )
            llm_answer = self.ollama.generate(
                prompt=prompt,
                system=system,
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
