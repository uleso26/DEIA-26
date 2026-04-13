# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import time

from agents.orchestrator import T2DOrchestrator


# Define the constants lookup tables and settings used below
DEMO_QUERIES = [
    "In tirzepatide-treated patients with established cardiovascular disease, which post-marketing FAERS safety signals merit closer review beyond the expected gastrointestinal profile?",
    "For weekly incretin selection, summarize the direct phase 3 HbA1c and weight efficacy gap between tirzepatide and semaglutide in SURPASS-2.",
    "For a patient with T2D, obesity, and persistent hyperglycaemia on metformin, how does the ADA 2025 guideline sequence the next step, and how does that differ from NICE?",
    "Give me a last 6 months evidence update on SGLT2 inhibitors in heart failure, prioritizing meta-analyses and large real-world studies.",
]


# Run the main workflow implemented by this module
def run() -> dict:
    orchestrator = T2DOrchestrator()
    results = []
    try:
        for query in DEMO_QUERIES:
            started = time.perf_counter()
            response = orchestrator.run_query(query)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            results.append(
                {
                    "query": query,
                    "question_class": response["question_class"],
                    "latency_ms": elapsed_ms,
                    "num_citations": len(response["citations"]),
                    "num_sections": len(response["sections"]),
                }
            )
    finally:
        orchestrator.close()

    average = round(sum(item["latency_ms"] for item in results) / len(results), 2) if results else 0.0
    return {"num_queries": len(results), "average_latency_ms": average, "results": results}


# CLI entrypoint
if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
