# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

from tools.native_tools import search_retrieval_index


# Define the constants lookup tables and settings used below
RETRIEVAL_CASES = [
    {
        "query": "SGLT2 inhibitors heart failure recent literature",
        "expected_refs": {"41597355", "41517628"},
    },
    {
        "query": "tirzepatide safety post-marketing surveillance",
        "expected_refs": {"40037695"},
    },
    {
        "query": "oral GLP-1 late-stage development",
        "expected_refs": {"41398455", "40960239"},
    },
]


# Run the main workflow implemented by this module
def run() -> dict:
    hits = 0
    reciprocal_rank_total = 0.0
    detailed = []

    for case in RETRIEVAL_CASES:
        results = search_retrieval_index(case["query"], top_k=3)
        retrieved_ids = [item.get("pmid") or item.get("id") or item.get("doc_id") for item in results]
        expected = case["expected_refs"]
        found = expected.intersection(retrieved_ids)
        if found:
            hits += 1
            best_rank = min(retrieved_ids.index(ref) + 1 for ref in found)
            reciprocal_rank_total += 1 / best_rank
        detailed.append(
            {
                "query": case["query"],
                "retrieved_ids": retrieved_ids,
                "expected_refs": sorted(expected),
                "hit": bool(found),
            }
        )

    count = len(RETRIEVAL_CASES)
    return {
        "num_queries": count,
        "hit_rate_at_3": round(hits / count, 4) if count else 0.0,
        "mrr_at_3": round(reciprocal_rank_total / count, 4) if count else 0.0,
        "details": detailed,
    }


# CLI entrypoint
if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
