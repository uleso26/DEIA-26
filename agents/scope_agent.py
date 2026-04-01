from __future__ import annotations

from core.models import AgentSection
from agents.base_agent import citation, unique_strings


CDC_TYPE2_URL = "https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html"
CDC_COMPLICATIONS_URL = "https://www.cdc.gov/diabetes/diabetes-complications/"
NHS_TYPE2_URL = "https://www.nhs.uk/conditions/type-2-diabetes/"


class ScopeAgent:
    def run(self, query: str, question_class: str) -> AgentSection:
        query_lower = query.lower()

        if question_class == "Q7":
            if any(term in query_lower for term in ["death", "die", "fatal", "dangerous", "serious", "scary", "life expectancy"]):
                summary = (
                    "Yes. Diabetes can be serious and can contribute to death, mainly through long-term complications "
                    "such as cardiovascular disease, stroke, kidney disease, infection, and acute metabolic emergencies "
                    "when it is untreated or poorly controlled. The level of risk is not the same for everyone, and good "
                    "monitoring and treatment can reduce it."
                )
            elif any(term in query_lower for term in ["complication", "complications", "affect kidneys", "affect eyes", "affect heart"]):
                summary = (
                    "Type 2 diabetes is a serious chronic disease because it can damage blood vessels and organs over time, "
                    "including the heart, kidneys, eyes, nerves, and feet, especially when glycaemic control remains poor."
                )
            else:
                summary = (
                    "Type 2 diabetes is a chronic disease with meaningful long-term health risks, but those risks vary by "
                    "the person, complications, and how well the condition is managed."
                )
            citations = [
                citation("CDC", "Type 2 Diabetes", "CDC:T2D", CDC_TYPE2_URL, "Tier 1", "2024-05-15"),
                citation("CDC", "Diabetes Complications", "CDC:COMPLICATIONS", CDC_COMPLICATIONS_URL, "Tier 1"),
            ]
            return AgentSection(
                agent="Scope Agent",
                question_class="Q7",
                summary=summary,
                citations=citations,
                caveats=[
                    "This is general medical information, not a personal risk assessment.",
                    "For personal symptoms or treatment decisions, use a clinician-led route.",
                ],
                evidence_tiers=unique_strings(["Tier 1"]),
                metadata={"mode": "disease_background"},
            )

        if question_class == "Q8":
            summary = (
                "This is a valid enterprise question, but this build does not include a live pricing, reimbursement, or "
                "formulary data feed. It can support clinical, safety, trial, mechanism, and competitor intelligence, but "
                "commercial access answers need market-specific pricing and payer data."
            )
            return AgentSection(
                agent="Scope Agent",
                question_class="Q8",
                summary=summary,
                caveats=["Pricing, reimbursement, and formulary status vary by market, date, and contract."],
                evidence_tiers=[],
                metadata={"mode": "market_access_scope"},
            )

        if question_class == "Q9":
            summary = (
                "This looks like a personal or urgent medical question. This platform is not designed for emergency triage "
                "or personal medical advice. If there is immediate danger or severe symptoms, seek urgent clinical care or "
                "call 911 now."
            )
            citations = [
                citation("NHS", "Type 2 diabetes", "NHS:T2D", NHS_TYPE2_URL, "Tier 1"),
            ]
            return AgentSection(
                agent="Scope Agent",
                question_class="Q9",
                summary=summary,
                citations=citations,
                caveats=["This platform does not provide emergency triage or personal medical advice."],
                evidence_tiers=unique_strings(["Tier 1"]),
                metadata={"mode": "urgent_guardrail"},
            )

        summary = (
            "This question is outside the current T2D enterprise intelligence scope. Rephrase it as a safety, trial, "
            "guideline, mechanism, competitor, literature or population, disease-background, or access-and-pricing question."
        )
        return AgentSection(
            agent="Scope Agent",
            question_class="Q0",
            summary=summary,
            citations=[],
            caveats=["The platform is optimized for T2D enterprise intelligence rather than arbitrary free-form tasks."],
            evidence_tiers=[],
            metadata={"mode": "out_of_scope"},
        )
