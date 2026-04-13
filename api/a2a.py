# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from core.models import FinalResponse
from core.runtime_utils import utc_now_iso


# Define the constants lookup tables and settings used below
PLATFORM_AGENT_KEY = "platform"
TRIAL_AGENT_KEY = "trial-evidence"
PUBLIC_AGENT_CARD_PATH = "/.well-known/agent-card.json"
A2A_TASK_LIMIT = 128


# Define the runtime protocol expected by the thin A2A facade
class A2ARuntime(Protocol):
    def run_query(self, query: str) -> dict[str, Any]:
        ...


# Define the A2A stream plan that describes one streamed request
@dataclass(frozen=True)
class A2AStreamPlan:
    agent_key: str
    request_payload: dict[str, Any]
    query: str
    base_url: str


# Keep a small in-memory task store for recent A2A request state
class _TaskStore:
    """Keep a small in-memory record of recent A2A task results."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def save(self, task: dict[str, Any]) -> None:
        task_id = str(task.get("id", ""))
        if not task_id:
            return
        with self._lock:
            self._tasks[task_id] = json.loads(json.dumps(task))
            while len(self._tasks) > A2A_TASK_LIMIT:
                self._tasks.popitem(last=False)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return json.loads(json.dumps(task)) if task else None

    def list_for_agent(self, agent_key: str) -> list[dict[str, Any]]:
        with self._lock:
            tasks = [task for task in self._tasks.values() if task.get("agentId") == _agent_id(agent_key)]
            return json.loads(json.dumps(tasks))

    def clear(self) -> None:
        with self._lock:
            self._tasks.clear()


_TASKS = _TaskStore()


# Clear A2A tasks from the in-memory runtime state
def clear_a2a_tasks() -> None:
    """Reset in-memory A2A task state for tests."""
    _TASKS.clear()


# Check whether A2A path applies to the current input
def is_a2a_path(path: str) -> bool:
    """Return True when the path belongs to the thin A2A facade."""
    normalized_path = urlsplit(path).path or path
    return normalized_path == PUBLIC_AGENT_CARD_PATH or normalized_path.startswith("/a2a/")


# Dispatch A2A request to the appropriate handler
def dispatch_a2a_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    runtime: A2ARuntime,
    base_url: str,
) -> tuple[int, dict[str, Any]] | None:
    """Handle non-stream A2A requests and return JSON payloads."""
    normalized_path = urlsplit(path).path or path
    if normalized_path == PUBLIC_AGENT_CARD_PATH:
        if method != "GET":
            return 405, {"ok": False, "error": "Unsupported method for public agent card."}
        return 200, build_agent_card(PLATFORM_AGENT_KEY, base_url)

    route = _match_a2a_route(normalized_path)
    if route is None:
        return None

    agent_key, action, task_id = route
    if action == "agent-card":
        if method != "GET":
            return 405, {"ok": False, "error": "Unsupported method for agent card."}
        return 200, build_agent_card(agent_key, base_url)
    if action == "extended-agent-card":
        if method != "GET":
            return 405, {"ok": False, "error": "Unsupported method for extended agent card."}
        return 200, build_agent_card(agent_key, base_url, extended=True)
    if action == "message:send":
        if method != "POST":
            return 405, {"ok": False, "error": "Unsupported method for message send."}
        return _handle_message_send(agent_key, payload or {}, runtime)
    if action == "tasks":
        if method != "GET":
            return 405, {"ok": False, "error": "Unsupported method for task listing."}
        return 200, {"tasks": _TASKS.list_for_agent(agent_key)}
    if action == "task-detail":
        if method != "GET":
            return 405, {"ok": False, "error": "Unsupported method for task lookup."}
        task = _TASKS.get(task_id or "")
        if not task or task.get("agentId") != _agent_id(agent_key):
            return 404, {"ok": False, "error": f"Unknown task: {task_id}"}
        return 200, {"task": task}
    if action == "message:stream":
        return 405, {"ok": False, "error": "Use POST streaming handler for message:stream."}

    return 404, {"ok": False, "error": f"Unknown A2A path: {normalized_path}"}


# Prepare A2A stream for the next execution step
def prepare_a2a_stream(
    path: str,
    payload: dict[str, Any] | None,
    runtime: A2ARuntime,
    base_url: str,
) -> A2AStreamPlan | None:
    """Return stream execution details when a POST targets an A2A stream endpoint."""
    normalized_path = urlsplit(path).path or path
    route = _match_a2a_route(normalized_path)
    if route is None:
        return None
    agent_key, action, _task_id = route
    if action != "message:stream":
        return None
    request_payload = payload or {}
    query = extract_message_text(request_payload)
    if not query:
        raise ValueError("A2A message payload must include a non-empty text part.")
    return A2AStreamPlan(agent_key=agent_key, request_payload=request_payload, query=query, base_url=base_url)


# Generate stream payloads from the current runtime state
def generate_stream_payloads(plan: A2AStreamPlan, runtime: A2ARuntime) -> Iterable[dict[str, Any]]:
    """Yield a small sequence of SSE-ready payloads for A2A streaming."""
    task_id = _new_task_id(plan.agent_key)
    context_id = _extract_context_id(plan.request_payload)
    working_task = _task_payload(
        agent_key=plan.agent_key,
        task_id=task_id,
        context_id=context_id,
        query=plan.query,
        state="working",
        response_payload=None,
    )
    _TASKS.save(working_task)
    yield {"task": working_task}

    try:
        response_payload = _run_a2a_agent(plan.agent_key, plan.query, runtime)
    except Exception as exc:
        failed_task = _task_payload(
            agent_key=plan.agent_key,
            task_id=task_id,
            context_id=context_id,
            query=plan.query,
            state="failed",
            response_payload=None,
            error=str(exc),
        )
        _TASKS.save(failed_task)
        yield {"task": failed_task}
        return

    answer = str(response_payload.get("answer", "")).strip()
    if answer:
        yield {
            "artifactUpdate": {
                "taskId": task_id,
                "artifact": {
                    "artifactId": f"{task_id}:answer",
                    "parts": [{"kind": "text", "text": answer}],
                },
            }
        }
    completed_task = _task_payload(
        agent_key=plan.agent_key,
        task_id=task_id,
        context_id=context_id,
        query=plan.query,
        state="completed",
        response_payload=response_payload,
    )
    _TASKS.save(completed_task)
    yield {"task": completed_task}


# Build agent card for the downstream execution path
def build_agent_card(agent_key: str, base_url: str, *, extended: bool = False) -> dict[str, Any]:
    """Build an A2A-compatible agent card for the requested agent surface."""
    if agent_key == PLATFORM_AGENT_KEY:
        name = "T2D Platform Orchestrator Agent"
        description = (
            "Top-level T2D enterprise intelligence agent. It orchestrates routing, evidence planning, "
            "specialist execution, review, and governed synthesis."
        )
        skills = [
            _skill(
                "t2d_enterprise_query",
                "Run a governed T2D enterprise query across the full platform.",
                ["routing", "governance", "multi-source evidence"],
                [
                    "What does SURPASS-3 show?",
                    "ADA pathway after metformin for obesity and CKD",
                ],
            ),
            _skill(
                "guideline_pathway_analysis",
                "Analyze ADA or NICE sequencing logic under stated clinical context.",
                ["guidelines", "sequencing", "clinical context"],
                ["First-line treatment for newly diagnosed T2D", "ADA versus NICE after metformin"],
            ),
            _skill(
                "safety_surveillance_summary",
                "Summarize T2D safety and label evidence with caveated interpretation.",
                ["safety", "labels", "pharmacovigilance"],
                ["FAERS safety review for tirzepatide"],
            ),
            _skill(
                "literature_population_summary",
                "Summarize literature updates and population burden evidence.",
                ["literature", "retrieval", "public health"],
                ["Last 6 months evidence update on SGLT2 inhibitors in heart failure"],
            ),
        ]
    elif agent_key == TRIAL_AGENT_KEY:
        name = "T2D Trial Evidence Agent"
        description = (
            "Specialist trial-evidence agent for study lookup, comparative efficacy, and publication-linked "
            "trial summaries."
        )
        skills = [
            _skill(
                "trial_lookup",
                "Retrieve structured detail for a named T2D clinical trial.",
                ["trials", "clinical evidence"],
                ["What does SURPASS-3 show?"],
            ),
            _skill(
                "trial_compare",
                "Compare trial evidence across two therapies or study programs.",
                ["comparative efficacy", "phase 3"],
                ["Compare tirzepatide and semaglutide trial evidence"],
            ),
            _skill(
                "trial_publication_summary",
                "Summarize the linked publication evidence for a named trial.",
                ["publications", "trial summaries"],
                ["SURPASS-3 publication summary"],
            ),
        ]
    else:
        raise ValueError(f"Unknown A2A agent: {agent_key}")

    card = {
        "id": _agent_id(agent_key),
        "name": name,
        "description": description,
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "url": f"{base_url}{_agent_base_path(agent_key)}/v1",
        "preferredTransport": "REST",
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text", "application/json"],
        "capabilities": {
            "streaming": True,
            "taskHistory": True,
            "authenticatedExtendedCard": True,
        },
        "supportedInterfaces": [
            {
                "transport": "HTTP",
                "protocolBinding": "REST",
                "url": f"{base_url}{_agent_base_path(agent_key)}/v1",
            }
        ],
        "skills": skills,
    }
    if extended:
        card["metadata"] = {
            "deliveryModel": "thin_a2a_facade",
            "internalOrchestration": "LangGraph",
            "tooling": ["MCP", "native LangChain tools"],
            "notes": (
                "This card exposes a minimal A2A-compatible interoperability surface. "
                "Internal execution remains centralized inside the platform runtime."
            ),
        }
    return card


# Extract message text from the upstream payload
def extract_message_text(payload: dict[str, Any]) -> str:
    """Extract a free-text request from a permissive A2A request body."""
    direct_query = payload.get("query") or payload.get("text")
    if isinstance(direct_query, str) and direct_query.strip():
        return direct_query.strip()

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        parts = message.get("parts")
        if isinstance(parts, list):
            text_parts = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("kind") == "text" and isinstance(part.get("text"), str):
                    text = part["text"].strip()
                    if text:
                        text_parts.append(text)
            if text_parts:
                return " ".join(text_parts)
    return ""


# Match A2A route against the supported routes or entities
def _match_a2a_route(path: str) -> tuple[str, str, str | None] | None:
    normalized_path = urlsplit(path).path or path
    if normalized_path in {
        "/a2a/platform/.well-known/agent-card.json",
        "/a2a/platform/v1/agent-card.json",
    }:
        return PLATFORM_AGENT_KEY, "agent-card", None
    if normalized_path == "/a2a/platform/v1/extended-agent-card.json":
        return PLATFORM_AGENT_KEY, "extended-agent-card", None
    if normalized_path == "/a2a/platform/v1/message:send":
        return PLATFORM_AGENT_KEY, "message:send", None
    if normalized_path == "/a2a/platform/v1/message:stream":
        return PLATFORM_AGENT_KEY, "message:stream", None
    if normalized_path == "/a2a/platform/v1/tasks":
        return PLATFORM_AGENT_KEY, "tasks", None
    if normalized_path.startswith("/a2a/platform/v1/tasks/"):
        return PLATFORM_AGENT_KEY, "task-detail", normalized_path.rsplit("/", 1)[-1]

    if normalized_path in {
        "/a2a/trial-evidence/.well-known/agent-card.json",
        "/a2a/trial-evidence/v1/agent-card.json",
    }:
        return TRIAL_AGENT_KEY, "agent-card", None
    if normalized_path == "/a2a/trial-evidence/v1/extended-agent-card.json":
        return TRIAL_AGENT_KEY, "extended-agent-card", None
    if normalized_path == "/a2a/trial-evidence/v1/message:send":
        return TRIAL_AGENT_KEY, "message:send", None
    if normalized_path == "/a2a/trial-evidence/v1/message:stream":
        return TRIAL_AGENT_KEY, "message:stream", None
    if normalized_path == "/a2a/trial-evidence/v1/tasks":
        return TRIAL_AGENT_KEY, "tasks", None
    if normalized_path.startswith("/a2a/trial-evidence/v1/tasks/"):
        return TRIAL_AGENT_KEY, "task-detail", normalized_path.rsplit("/", 1)[-1]
    return None


# Handle message send within the current service surface
def _handle_message_send(
    agent_key: str,
    payload: dict[str, Any],
    runtime: A2ARuntime,
) -> tuple[int, dict[str, Any]]:
    query = extract_message_text(payload)
    if not query:
        return 400, {"ok": False, "error": "A2A message payload must include a non-empty text part."}
    task_id = _new_task_id(agent_key)
    context_id = _extract_context_id(payload)
    try:
        response_payload = _run_a2a_agent(agent_key, query, runtime)
    except Exception as exc:  # pragma: no cover - defensive HTTP surface
        failed_task = _task_payload(
            agent_key=agent_key,
            task_id=task_id,
            context_id=context_id,
            query=query,
            state="failed",
            response_payload=None,
            error=str(exc),
        )
        _TASKS.save(failed_task)
        return 500, {"task": failed_task}

    task = _task_payload(
        agent_key=agent_key,
        task_id=task_id,
        context_id=context_id,
        query=query,
        state="completed",
        response_payload=response_payload,
    )
    _TASKS.save(task)
    return 200, {"task": task}


# Run the selected A2A agent and normalize its response metadata
def _run_a2a_agent(agent_key: str, query: str, runtime: A2ARuntime) -> dict[str, Any]:
    if agent_key == PLATFORM_AGENT_KEY:
        response_payload = runtime.run_query(query)
    elif agent_key == TRIAL_AGENT_KEY:
        response_payload = _run_trial_specialist(query, runtime)
    else:  # pragma: no cover - route matching protects this
        raise ValueError(f"Unknown A2A agent: {agent_key}")
    metadata = response_payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("delivery_mode", "a2a")
        metadata.setdefault("a2a_agent_id", _agent_id(agent_key))
    return response_payload


# Run the trial specialist surface and synthesize its final answer
def _run_trial_specialist(query: str, runtime: A2ARuntime) -> dict[str, Any]:
    trial_agent = getattr(runtime, "trial_agent", None)
    synthesis_agent = getattr(runtime, "synthesis_agent", None)
    if trial_agent is None or synthesis_agent is None:
        return runtime.run_query(query)

    section = trial_agent.run(query)
    trace_id = f"a2a-trial-{uuid4().hex[:12]}"
    workflow_metadata = {
        "delivery_mode": "a2a",
        "a2a_agent_id": _agent_id(TRIAL_AGENT_KEY),
        "execution_surface": "trial_specialist",
    }
    final_response = synthesis_agent.run("Q2", query, [section], trace_id, workflow_metadata=workflow_metadata)
    if isinstance(final_response, FinalResponse):
        return final_response.to_dict()
    return final_response


# Build the persisted task payload returned by the A2A facade
def _task_payload(
    *,
    agent_key: str,
    task_id: str,
    context_id: str,
    query: str,
    state: str,
    response_payload: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    artifacts = _artifacts_for_response(task_id, response_payload) if response_payload else []
    status_text = {
        "working": "Processing request.",
        "completed": "Completed successfully.",
        "failed": error or "Task failed.",
    }[state]
    task = {
        "id": task_id,
        "kind": "task",
        "agentId": _agent_id(agent_key),
        "contextId": context_id,
        "createdAt": utc_now_iso(),
        "status": {
            "state": f"TASK_STATE_{state.upper()}",
            "message": {
                "role": "agent",
                "parts": [{"kind": "text", "text": status_text}],
            },
        },
        "artifacts": artifacts,
        "metadata": {
            "query": query,
            "agentKey": agent_key,
            "responseQuestionClass": (response_payload or {}).get("question_class"),
        },
    }
    if error:
        task["error"] = {"message": error}
    return task


# Build response artifacts for completed A2A task payloads
def _artifacts_for_response(task_id: str, response_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "artifactId": f"{task_id}:answer",
            "name": "answer",
            "parts": [{"kind": "text", "text": str(response_payload.get("answer", ""))}],
            "metadata": {
                "questionClass": response_payload.get("question_class"),
                "traceId": response_payload.get("trace_id"),
            },
        },
        {
            "artifactId": f"{task_id}:payload",
            "name": "final_response",
            "parts": [{"kind": "data", "data": response_payload}],
        },
    ]


# Build a declared A2A skill record for an agent card
def _skill(skill_id: str, description: str, tags: list[str], examples: list[str]) -> dict[str, Any]:
    return {
        "id": skill_id,
        "name": skill_id.replace("_", " ").title(),
        "description": description,
        "tags": tags,
        "examples": examples,
    }


# Build the stable A2A agent identifier for a given agent key
def _agent_id(agent_key: str) -> str:
    return {
        PLATFORM_AGENT_KEY: "t2d-platform-orchestrator-agent",
        TRIAL_AGENT_KEY: "t2d-trial-evidence-agent",
    }[agent_key]


# Build the base HTTP path for the selected A2A agent
def _agent_base_path(agent_key: str) -> str:
    return {
        PLATFORM_AGENT_KEY: "/a2a/platform",
        TRIAL_AGENT_KEY: "/a2a/trial-evidence",
    }[agent_key]


# Generate a new task identifier for the A2A facade
def _new_task_id(agent_key: str) -> str:
    return f"{agent_key}-{uuid4().hex[:12]}"


# Extract the context identifier from an A2A task payload
def _extract_context_id(payload: dict[str, Any]) -> str:
    context_id = payload.get("contextId")
    if isinstance(context_id, str) and context_id.strip():
        return context_id.strip()
    message = payload.get("message")
    if isinstance(message, dict):
        message_id = message.get("messageId")
        if isinstance(message_id, str) and message_id.strip():
            return f"context-{message_id.strip()}"
    return f"context-{uuid4().hex[:10]}"
