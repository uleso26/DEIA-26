from __future__ import annotations

import argparse
import json

from agents.orchestrator import T2DOrchestrator, bootstrap_runtime
from api.http_server import serve_http_api
from core.storage import backend_status
from evaluation.groundedness_eval import run as groundedness_eval
from evaluation.latency_eval import run as latency_eval
from evaluation.retrieval_eval import run as retrieval_eval
from evaluation.routing_eval import run as routing_eval


def print_response(payload: dict) -> None:
    question_class_name = payload.get("metadata", {}).get("question_class_name")
    if question_class_name:
        print(f"Question Class: {payload['question_class']} ({question_class_name})")
    else:
        print(f"Question Class: {payload['question_class']}")
    print(f"Answer: {payload['answer']}")
    if payload["caveats"]:
        print("Notes:")
        for caveat in payload["caveats"]:
            print(f"- {caveat}")
    if payload["citations"]:
        print("Citations:")
        for citation in payload["citations"]:
            print(f"- {citation['reference_id']}: {citation['title']} ({citation['source']})")
    print(f"Trace ID: {payload['trace_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="T2D Therapeutic Intelligence Platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Build all local seed data and storage artifacts.")
    bootstrap_parser.add_argument("--sync-mongodb", action="store_true", default=None, help="Sync processed collections into MongoDB.")
    bootstrap_parser.add_argument("--sync-neo4j", action="store_true", default=None, help="Sync graph data into Neo4j.")

    query_parser = subparsers.add_parser("query", help="Run a query through the orchestrator.")
    query_parser.add_argument("text", help="The user query to run.")
    query_parser.add_argument("--json", action="store_true", help="Print raw JSON response.")

    subparsers.add_parser("backend-status", help="Report whether live MongoDB and Neo4j backends are reachable.")

    serve_parser = subparsers.add_parser("serve", help="Run a lightweight HTTP API for queries and health checks.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind.")

    eval_parser = subparsers.add_parser("eval", help="Run an evaluation suite.")
    eval_parser.add_argument("suite", choices=["routing", "groundedness", "latency", "retrieval"])

    args = parser.parse_args()

    if args.command == "bootstrap":
        bootstrap_runtime(sync_to_mongodb=args.sync_mongodb, sync_to_neo4j=args.sync_neo4j)
        print("Bootstrap complete.")
        return

    if args.command == "query":
        orchestrator = T2DOrchestrator()
        try:
            response = orchestrator.run_query(args.text)
        finally:
            orchestrator.close()
        if args.json:
            print(json.dumps(response, indent=2))
        else:
            print_response(response)
        return

    if args.command == "backend-status":
        print(json.dumps(backend_status(), indent=2))
        return

    if args.command == "serve":
        serve_http_api(host=args.host, port=args.port)
        return

    if args.command == "eval":
        if args.suite == "routing":
            print(json.dumps(routing_eval(), indent=2))
        elif args.suite == "groundedness":
            print(json.dumps(groundedness_eval(), indent=2))
        elif args.suite == "latency":
            print(json.dumps(latency_eval(), indent=2))
        else:
            print(json.dumps(retrieval_eval(), indent=2))


if __name__ == "__main__":
    main()
