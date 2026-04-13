# Imports.
from __future__ import annotations

from agents.orchestrator import T2DOrchestrator


# Module constants.
DEMO_QUERIES = [
    "What adverse event signals are emerging for tirzepatide in cardiac patients?",
    "Compare HbA1c reduction across Phase 3 trials for semaglutide vs tirzepatide",
    "What does SURPASS-3 show?",
    "For a T2D patient with CKD stage 3, what does NICE guidance suggest after metformin failure?",
    "ADA pathway after metformin for obesity",
    "Which drugs share the GLP1R mechanism?",
    "Give me an evidence update on SGLT2 inhibitors in heart failure",
    "What is the diabetes prevalence in the United Kingdom?",
]


# Check response.
def _check_response(response: dict) -> list[str]:
    issues = []
    if not response.get("citations"):
        issues.append("Missing citations")
    if not response.get("caveats"):
        issues.append("Missing caveats")
    if not response.get("evidence_tiers"):
        issues.append("Missing evidence tiers")
    if not response.get("trace_id"):
        issues.append("Missing trace ID")

    question_class = response["question_class"]
    answer = response["answer"]
    caveats = " ".join(response["caveats"])

    if question_class == "Q1" and "causality" not in caveats.lower():
        issues.append("Q1 missing non-causality caveat")
    if question_class == "Q2" and "comparison" not in answer.lower():
        issues.append("Q2 missing comparison wording")
    if question_class == "Q3" and "not medical advice" not in answer.lower():
        issues.append("Q3 missing guideline disclaimer")
    if question_class == "Q6" and not response.get("metadata", {}).get("population_context_used") and "recent evidence includes" not in answer.lower():
        issues.append("Q6 missing recency framing")
    if response.get("metadata", {}).get("population_context_used") and "population context indicates" not in answer.lower():
        issues.append("Population-context query missing surveillance framing")
    return issues


# Check query specific expectations.
def _check_query_specific_expectations(query: str, response: dict) -> list[str]:
    issues = []
    answer = response["answer"]
    if query == "What does SURPASS-3 show?" and "surpass-3" not in answer.lower():
        issues.append("SURPASS-3 query did not return SURPASS-3 content")
    if query == "ADA pathway after metformin for obesity":
        if "ada" not in answer.lower() and "glp-1 receptor agonist" not in answer.lower():
            issues.append("ADA obesity pathway query did not prioritize the ADA/GLP-1 branch")
        if "nice" in answer.lower() and "ada" not in answer.lower():
            issues.append("ADA obesity pathway query incorrectly returned only NICE framing")
    if query == "Which drugs share the GLP1R mechanism?":
        for expected in ("semaglutide", "tirzepatide"):
            if expected not in answer.lower():
                issues.append(f"GLP1R mechanism answer missing {expected}")
    if query == "Give me an evidence update on SGLT2 inhibitors in heart failure" and "oral glp-1 agonists" in answer.lower():
        issues.append("SGLT2 evidence update included an irrelevant oral GLP-1 paper")
    if query == "What is the diabetes prevalence in the United Kingdom?":
        if "united kingdom" not in answer.lower():
            issues.append("Population query did not mention the requested country")
        if "percent" not in answer.lower():
            issues.append("Population query did not surface the WHO prevalence value")
    return issues


# Run.
def run() -> dict:
    orchestrator = T2DOrchestrator()
    results = []
    try:
        for query in DEMO_QUERIES:
            response = orchestrator.run_query(query)
            issues = _check_response(response)
            issues.extend(_check_query_specific_expectations(query, response))
            results.append(
                {
                    "query": query,
                    "question_class": response["question_class"],
                    "issues": issues,
                    "passed": not issues,
                }
            )
    finally:
        orchestrator.close()

    passed = sum(1 for result in results if result["passed"])
    return {
        "num_queries": len(results),
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "results": results,
    }


# CLI entrypoint.
if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
