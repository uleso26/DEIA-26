# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

import pytest

from api.a2a import clear_a2a_tasks
from api.http_server import create_http_server, dispatch_request
from core.models import AgentSection, FinalResponse


pytestmark = pytest.mark.unit


# Provide a lightweight trial agent stub for this test module
class StubTrialAgent:
    def run(self, query: str) -> AgentSection:
        return AgentSection(
            agent="Clinical Trial Agent",
            question_class="Q2",
            summary=f"trial-section:{query}",
            citations=[],
            caveats=[],
            evidence_tiers=["Tier 2"],
            metadata={"comparison_type": "direct"},
        )


# Provide a lightweight synthesis agent stub for this test module
class StubSynthesisAgent:
    def run(
        self,
        question_class: str,
        query: str,
        sections: list[AgentSection],
        trace_id: str,
        workflow_metadata: dict[str, object] | None = None,
    ) -> FinalResponse:
        return FinalResponse(
            question_class=question_class,
            answer=f"trial-specialist:{query}",
            citations=[],
            caveats=[],
            evidence_tiers=["Tier 2"],
            trace_id=trace_id,
            sections=sections,
            metadata=dict(workflow_metadata or {}),
        )


# Provide a lightweight runtime stub for this test module
class StubRuntime:
    def __init__(self) -> None:
        self.trial_agent = StubTrialAgent()
        self.synthesis_agent = StubSynthesisAgent()

    def run_query(self, query: str) -> dict[str, object]:
        return {
            "question_class": "Q3",
            "answer": f"platform:{query}",
            "citations": [],
            "caveats": [],
            "evidence_tiers": [],
            "trace_id": "trace-test",
            "sections": [],
            "metadata": {},
        }

    def close(self) -> None:
        return


# Reset the in-memory A2A task store between tests
@pytest.fixture(autouse=True)
def reset_a2a_tasks() -> None:
    clear_a2a_tasks()


# Verify public agent card exposes platform skills
def test_public_agent_card_exposes_platform_skills() -> None:
    status_code, payload = dispatch_request(
        "GET",
        "/.well-known/agent-card.json",
        None,
        StubRuntime(),
        base_url="http://localhost:8000",
    )
    assert status_code == 200
    assert payload["id"] == "t2d-platform-orchestrator-agent"
    assert payload["url"] == "http://localhost:8000/a2a/platform/v1"
    assert any(skill["id"] == "t2d_enterprise_query" for skill in payload["skills"])


# Verify platform A2A send returns completed task and stores it
def test_platform_a2a_send_returns_completed_task_and_stores_it() -> None:
    runtime = StubRuntime()
    status_code, payload = dispatch_request(
        "POST",
        "/a2a/platform/v1/message:send",
        {
            "message": {
                "messageId": "msg-platform",
                "role": "user",
                "parts": [{"kind": "text", "text": "What does SURPASS-3 show?"}],
            }
        },
        runtime,
        base_url="http://localhost:8000",
    )
    assert status_code == 200
    task = payload["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["text"] == "platform:What does SURPASS-3 show?"

    task_id = task["id"]
    status_code, task_lookup = dispatch_request(
        "GET",
        f"/a2a/platform/v1/tasks/{task_id}",
        None,
        runtime,
        base_url="http://localhost:8000",
    )
    assert status_code == 200
    assert task_lookup["task"]["id"] == task_id


# Verify trial evidence A2A send uses specialist trial surface
def test_trial_evidence_a2a_send_uses_specialist_trial_surface() -> None:
    status_code, payload = dispatch_request(
        "POST",
        "/a2a/trial-evidence/v1/message:send",
        {
            "message": {
                "messageId": "msg-trial",
                "role": "user",
                "parts": [{"kind": "text", "text": "Compare tirzepatide and semaglutide"}],
            }
        },
        StubRuntime(),
        base_url="http://localhost:8000",
    )
    assert status_code == 200
    task = payload["task"]
    response_payload = task["artifacts"][1]["parts"][0]["data"]
    assert response_payload["question_class"] == "Q2"
    assert response_payload["answer"] == "trial-specialist:Compare tirzepatide and semaglutide"
    assert response_payload["metadata"]["a2a_agent_id"] == "t2d-trial-evidence-agent"


# Verify platform A2A stream returns SSE task updates
def test_platform_a2a_stream_returns_sse_task_updates() -> None:
    runtime = StubRuntime()
    try:
        server = create_http_server("127.0.0.1", 0, runtime)
    except PermissionError as exc:
        pytest.skip(f"Socket binding is not permitted in this environment: {exc}")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address
    request = Request(
        f"http://{host}:{port}/a2a/platform/v1/message:stream",
        data=json.dumps(
            {
                "message": {
                    "messageId": "msg-stream",
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Give me a pathway summary"}],
                }
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
        assert '"TASK_STATE_WORKING"' in body
        assert '"artifactUpdate"' in body
        assert "platform:Give me a pathway summary" in body
        assert '"TASK_STATE_COMPLETED"' in body
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
