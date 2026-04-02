from __future__ import annotations

import sys

from core.models import AgentSection


ROUTER_SYSTEM_TEMPLATE = (
    "You are a strict diabetes-enterprise query router. Output one label only: Q1, Q2, Q3, Q4, Q5, or Q6."
)
ROUTER_HUMAN_TEMPLATE = (
    "Classify the diabetes intelligence request into exactly one label.\n"
    "Q1=safety surveillance\n"
    "Q2=trial efficacy comparison or trial detail\n"
    "Q3=guideline pathway or treatment sequencing\n"
    "Q4=target or mechanism landscape\n"
    "Q5=pipeline or competitor monitoring\n"
    "Q6=literature or population evidence update\n"
    "Query: {query}\n"
    "Return only the label."
)
SYNTHESIS_SYSTEM_TEMPLATE = "You synthesize grounded diabetes evidence into concise clinical intelligence answers."
SYNTHESIS_HUMAN_TEMPLATE = (
    "Question class: {question_class}\n"
    "User query: {query}\n"
    "Evidence summaries:\n{section_block}\n\n"
    "Write one grounded answer in one or two sentences. Use only the supplied evidence. "
    "If guideline branches differ, say so explicitly. Do not invent facts."
)


def _supports_langchain_prompt_rendering() -> bool:
    # LangChain prompt rendering works cleanly in the main 3.11 runtime used by
    # the platform, but the local Anaconda 3.13 pytest interpreter pulls a much
    # heavier optional ML stack. Keep that path guarded so tests stay portable.
    return sys.version_info < (3, 13)


def _render_with_langchain(system_template: str, human_template: str, **kwargs: str) -> tuple[str, str] | None:
    if not _supports_langchain_prompt_rendering():
        return None
    try:
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([("system", system_template), ("human", human_template)])
        prompt_value = template.invoke(kwargs)
        messages = prompt_value.to_messages()
        system_parts: list[str] = []
        prompt_parts: list[str] = []
        for message in messages:
            if message.type == "system":
                system_parts.append(str(message.content))
            else:
                prompt_parts.append(str(message.content))
        return "\n\n".join(system_parts).strip(), "\n\n".join(prompt_parts).strip()
    except Exception:
        return None


def render_ollama_messages(system_template: str, human_template: str, **kwargs: str) -> tuple[str, str]:
    rendered = _render_with_langchain(system_template, human_template, **kwargs)
    if rendered:
        return rendered
    return system_template, human_template.format(**kwargs)


def section_block_for_prompt(sections: list[AgentSection]) -> str:
    return "\n\n".join(
        [
            f"Agent: {section.agent}\nSummary: {section.summary}\nCaveats: {'; '.join(section.caveats)}"
            for section in sections
            if section.summary
        ]
    )
