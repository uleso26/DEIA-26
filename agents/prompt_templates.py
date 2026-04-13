# Imports.
from __future__ import annotations

import sys

from core.models import AgentSection


# Module constants.
ROUTER_SYSTEM_TEMPLATE = (
    "You are a strict diabetes-enterprise query router. "
    "Scoped routes such as Q0, Q7, Q8, and Q9 are handled before you run. "
    "Output one label only: Q1, Q2, Q3, Q4, Q5, or Q6."
)
ROUTER_HUMAN_TEMPLATE = (
    "Classify the diabetes intelligence request into exactly one label.\n"
    "Only enterprise-core lanes are valid here because scope guardrails are handled upstream.\n"
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
PLANNER_SYSTEM_TEMPLATE = (
    "You are a bounded evidence planner for a diabetes intelligence system. "
    "Choose the smallest valid execution plan from the allowed nodes only. "
    "Return JSON only."
)
PLANNER_HUMAN_TEMPLATE = (
    "Query: {query}\n"
    "Question class: {question_class}\n"
    "Resolved entities: {entities}\n"
    "Default execution nodes: {default_nodes}\n"
    "Allowed execution nodes: {allowed_nodes}\n"
    "Available native LangChain tools: {native_tools}\n"
    "Return JSON with keys execution_nodes (array of node names in order) and reason (short string)."
)
PLAN_REFINEMENT_SYSTEM_TEMPLATE = (
    "You are a bounded ReAct-style planner. Based on the evidence review, decide whether one more allowed node "
    "should be added. Return JSON only."
)
PLAN_REFINEMENT_HUMAN_TEMPLATE = (
    "Query: {query}\n"
    "Evidence review: {evidence_review}\n"
    "Current evidence summaries:\n{current_summaries}\n"
    "Available nodes to add: {available_nodes}\n"
    "Return JSON with keys add_nodes (array with zero or one node names) and reason."
)


# Supports LangChain prompt rendering.
def _supports_langchain_prompt_rendering() -> bool:
    # LangChain prompt rendering is optional here. The main 3.11 runtime works
    # cleanly, but the local Anaconda 3.13 pytest interpreter imports a heavier
    # optional ML stack. Keep the guard explicit so 3.13 falls back to plain
    # string formatting instead of failing silently during tests.
    return sys.version_info < (3, 13)


# Render with LangChain.
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


# Render Ollama messages.
def render_ollama_messages(system_template: str, human_template: str, **kwargs: str) -> tuple[str, str]:
    rendered = _render_with_langchain(system_template, human_template, **kwargs)
    if rendered:
        return rendered
    return system_template, human_template.format(**kwargs)


# Section block for prompt.
def section_block_for_prompt(sections: list[AgentSection]) -> str:
    return "\n\n".join(
        [
            f"Agent: {section.agent}\nSummary: {section.summary}\nCaveats: {'; '.join(section.caveats)}"
            for section in sections
            if section.summary
        ]
    )
