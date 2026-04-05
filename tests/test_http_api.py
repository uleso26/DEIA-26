from __future__ import annotations

import threading
from urllib.request import urlopen

import pytest

from api.http_server import create_http_server, dispatch_request, resolve_static_asset, stream_answer_chunks


pytestmark = pytest.mark.unit


class StubRuntime:
    def run_query(self, query: str) -> dict[str, object]:
        return {
            "question_class": "Q2",
            "answer": f"stub:{query}",
            "citations": [],
            "caveats": [],
            "evidence_tiers": [],
            "trace_id": "trace-test",
            "sections": [],
            "metadata": {},
        }

    def close(self) -> None:
        return


def test_http_api_query_endpoint_returns_json() -> None:
    status_code, payload = dispatch_request(
        "POST",
        "/query",
        {"query": "What does SURPASS-3 show?"},
        StubRuntime(),
    )
    assert status_code == 200
    assert payload["question_class"] == "Q2"
    assert payload["answer"] == "stub:What does SURPASS-3 show?"


def test_http_api_health_endpoint_returns_ok() -> None:
    status_code, payload = dispatch_request("GET", "/health", None, StubRuntime())
    assert status_code == 200
    assert payload == {"ok": True}


def test_http_api_ignores_query_string_on_known_path() -> None:
    status_code, payload = dispatch_request("GET", "/health?full=true", None, StubRuntime())
    assert status_code == 200
    assert payload == {"ok": True}


def test_static_homepage_asset_resolves() -> None:
    resolved = resolve_static_asset("/")
    assert resolved is not None
    asset_path, content_type = resolved
    assert asset_path.name == "index.html"
    assert content_type == "text/html"


def test_static_asset_blocks_traversal_path() -> None:
    assert resolve_static_asset("/static/../README.md") is None


def test_stream_answer_chunks_splits_answer_into_sse_sized_segments() -> None:
    chunks = stream_answer_chunks(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    )
    assert len(chunks) == 2
    assert chunks[0].endswith(" ")
    assert chunks[1] == "fifteen"


def test_sse_query_endpoint_streams_final_payload() -> None:
    runtime = StubRuntime()
    try:
        server = create_http_server("127.0.0.1", 0, runtime)
    except PermissionError as exc:
        pytest.skip(f"Socket binding is not permitted in this environment: {exc}")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address
    try:
        with urlopen(f"http://{host}:{port}/query/stream?query=test", timeout=5) as response:
            body = response.read().decode("utf-8")
        assert "event: status" in body
        assert "event: delta" in body
        assert "event: final" in body
        assert "stub:test" in body
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
