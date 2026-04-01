from __future__ import annotations

from collections import defaultdict

from agents.router_agent import RouterAgent
from core.paths import ROOT
from core.storage import load_json


def run() -> dict:
    queries = load_json(ROOT / "evaluation" / "test_queries.json")
    router = RouterAgent()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct = 0
    results = []
    for item in queries:
        predicted = router.route(item["query"])["question_class"]
        expected = item["expected_class"]
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
        results.append({"id": item["id"], "expected": expected, "predicted": predicted})
    accuracy = correct / len(queries) if queries else 0.0
    return {
        "num_queries": len(queries),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "confusion_matrix": {expected: dict(predicted) for expected, predicted in confusion.items()},
        "results": results,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
