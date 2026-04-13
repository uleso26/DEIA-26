# Imports.
from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from agents.orchestrator import T2DOrchestrator
from api.a2a import dispatch_a2a_request, generate_stream_payloads, is_a2a_path, prepare_a2a_stream
from core.storage import backend_status


# Query Runtime.
class QueryRuntime(Protocol):
    def run_query(self, query: str) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


# Module constants.
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REQUEST_BYTES = 1_000_000
SSE_WORD_CHUNK_SIZE = 14


# Stream answer chunks.
def stream_answer_chunks(answer: str, chunk_size_words: int = SSE_WORD_CHUNK_SIZE) -> list[str]:
    """Split an answer into small word chunks for local SSE streaming."""
    words = answer.split()
    if not words:
        return []
    chunks: list[str] = []
    for index in range(0, len(words), chunk_size_words):
        suffix = " " if index + chunk_size_words < len(words) else ""
        chunks.append(" ".join(words[index : index + chunk_size_words]) + suffix)
    return chunks


# Platform HTTP Request Handler.
class PlatformHTTPRequestHandler(BaseHTTPRequestHandler):
    """Small local HTTP surface for demo use; auth and rate limiting are out of scope."""

    runtime: QueryRuntime

    def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - quiet local server
        return

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status_code, body, "application/json")

    def _send_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # The local demo SSE endpoint emits a finite sequence of events per query.
        # Closing the connection after the final event avoids hanging clients and CI tests.
        self.send_header("Connection", "close")
        self.end_headers()

    def _send_sse_data(self, payload: dict[str, Any]) -> None:
        body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _send_sse_event(self, event_name: str, payload: dict[str, Any]) -> None:
        body = f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body exceeds {MAX_REQUEST_BYTES} bytes.")
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw_body.decode("utf-8") or "{}")

    def _base_url(self) -> str:
        host = self.headers.get("Host")
        if not host:
            server_host, server_port = self.server.server_address
            host = f"{server_host}:{server_port}"
        return f"http://{host}"

    def do_GET(self) -> None:  # noqa: N802
        normalized_path = urlsplit(self.path).path or self.path
        if normalized_path == "/query/stream":
            self._stream_query()
            return
        if is_a2a_path(self.path):
            status_code, payload = dispatch_request("GET", self.path, None, self.runtime, base_url=self._base_url())
            self._send_json(status_code, payload)
            return
        asset = resolve_static_asset(self.path)
        if asset:
            asset_path, content_type = asset
            self._send_bytes(200, asset_path.read_bytes(), content_type)
            return
        status_code, payload = dispatch_request("GET", self.path, None, self.runtime, base_url=self._base_url())
        self._send_json(status_code, payload)

    def do_HEAD(self) -> None:  # noqa: N802
        normalized_path = urlsplit(self.path).path or self.path
        if normalized_path == "/query/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            return
        asset = resolve_static_asset(self.path)
        if asset:
            asset_path, content_type = asset
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(asset_path.stat().st_size))
            self.end_headers()
            return
        status_code, payload = dispatch_request("GET", self.path, None, self.runtime, base_url=self._base_url())
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON body: {exc}"})
            return
        if is_a2a_path(self.path):
            try:
                stream_plan = prepare_a2a_stream(self.path, payload, self.runtime, self._base_url())
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            if stream_plan is not None:
                self._send_sse_headers()
                for event_payload in generate_stream_payloads(stream_plan, self.runtime):
                    self._send_sse_data(event_payload)
                self.close_connection = True
                return
        status_code, response = dispatch_request("POST", self.path, payload, self.runtime, base_url=self._base_url())
        self._send_json(status_code, response)

    def _stream_query(self) -> None:
        params = parse_qs(urlsplit(self.path).query)
        query = (params.get("query") or [""])[0].strip()
        if not query:
            self._send_json(400, {"ok": False, "error": "Query string parameter 'query' is required."})
            return

        self._send_sse_headers()
        self._send_sse_event("status", {"message": "Understanding query"})
        try:
            payload = self.runtime.run_query(query)
        except Exception as exc:  # pragma: no cover - defensive HTTP path
            self._send_sse_event("error", {"error": str(exc)})
            self.close_connection = True
            return

        answer = str(payload.get("answer", ""))
        chunks = stream_answer_chunks(answer)
        if not chunks:
            self._send_sse_event("final", payload)
            self.close_connection = True
            return
        for chunk in chunks:
            self._send_sse_event("delta", {"text": chunk})
        self._send_sse_event("final", payload)
        self.close_connection = True


# Dispatch request.
def dispatch_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    runtime: QueryRuntime,
    *,
    base_url: str = "http://127.0.0.1",
) -> tuple[int, dict[str, Any]]:
    """Handle an API request without requiring a bound socket."""
    normalized_path = urlsplit(path).path or path
    a2a_response = dispatch_a2a_request(method, path, payload, runtime, base_url=base_url)
    if a2a_response is not None:
        return a2a_response
    if method == "GET":
        if normalized_path == "/health":
            return 200, {"ok": True}
        if normalized_path == "/backend-status":
            return 200, backend_status()
        return 404, {"ok": False, "error": f"Unknown path: {normalized_path}"}

    if method == "POST":
        if normalized_path != "/query":
            return 404, {"ok": False, "error": f"Unknown path: {normalized_path}"}
        query = (payload or {}).get("query") or (payload or {}).get("text")
        if not isinstance(query, str) or not query.strip():
            return 400, {"ok": False, "error": "JSON body must include a non-empty 'query' string."}
        try:
            return 200, runtime.run_query(query)
        except Exception as exc:  # pragma: no cover - defensive HTTP path
            return 500, {"ok": False, "error": str(exc)}

    return 405, {"ok": False, "error": f"Unsupported method: {method}"}


# Resolve static asset.
def resolve_static_asset(path: str) -> tuple[Path, str] | None:
    normalized_path = urlsplit(path).path or path
    static_root = STATIC_DIR.resolve()
    if normalized_path == "/":
        asset_path = STATIC_DIR / "index.html"
    elif normalized_path.startswith("/static/"):
        relative_path = normalized_path.removeprefix("/static/")
        if not relative_path or any(part in {"..", ""} for part in Path(relative_path).parts):
            return None
        asset_path = STATIC_DIR / relative_path
    else:
        return None

    try:
        resolved_asset_path = asset_path.resolve()
    except FileNotFoundError:
        return None
    if not resolved_asset_path.is_relative_to(static_root):
        return None
    if not resolved_asset_path.exists() or not resolved_asset_path.is_file():
        return None
    content_type = mimetypes.guess_type(resolved_asset_path.name)[0] or "application/octet-stream"
    return resolved_asset_path, content_type


# Create HTTP server.
def create_http_server(host: str, port: int, runtime: QueryRuntime) -> ThreadingHTTPServer:
    """Create a local HTTP server bound to a supplied orchestrator-like runtime."""
    class PlatformHTTPServer(ThreadingHTTPServer):
        daemon_threads = True

    class Handler(PlatformHTTPRequestHandler):
        pass

    Handler.runtime = runtime
    return PlatformHTTPServer((host, port), Handler)


# Serve HTTP API.
def serve_http_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the local HTTP API until interrupted."""
    runtime = T2DOrchestrator()
    server = create_http_server(host, port, runtime)
    print(f"HTTP API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - CLI shutdown path
        pass
    finally:
        server.server_close()
        runtime.close()
