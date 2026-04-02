from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from agents.orchestrator import T2DOrchestrator
from core.storage import backend_status


class QueryRuntime(Protocol):
    def run_query(self, query: str) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


STATIC_DIR = Path(__file__).resolve().parent / "static"


class PlatformHTTPRequestHandler(BaseHTTPRequestHandler):
    """Small HTTP surface for local deployment and demonstration."""

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

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw_body.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        asset = resolve_static_asset(self.path)
        if asset:
            asset_path, content_type = asset
            self._send_bytes(200, asset_path.read_bytes(), content_type)
            return
        status_code, payload = dispatch_request("GET", self.path, None, self.runtime)
        self._send_json(status_code, payload)

    def do_HEAD(self) -> None:  # noqa: N802
        asset = resolve_static_asset(self.path)
        if asset:
            asset_path, content_type = asset
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(asset_path.stat().st_size))
            self.end_headers()
            return
        status_code, payload = dispatch_request("GET", self.path, None, self.runtime)
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
        status_code, response = dispatch_request("POST", self.path, payload, self.runtime)
        self._send_json(status_code, response)


def dispatch_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    runtime: QueryRuntime,
) -> tuple[int, dict[str, Any]]:
    """Handle an API request without requiring a bound socket."""
    normalized_path = urlsplit(path).path or path
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


def resolve_static_asset(path: str) -> tuple[Path, str] | None:
    normalized_path = urlsplit(path).path or path
    if normalized_path == "/":
        asset_path = STATIC_DIR / "index.html"
    elif normalized_path.startswith("/static/"):
        relative_path = normalized_path.removeprefix("/static/")
        if not relative_path or any(part in {"..", ""} for part in Path(relative_path).parts):
            return None
        asset_path = STATIC_DIR / relative_path
    else:
        return None

    if not asset_path.exists() or not asset_path.is_file():
        return None
    content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    return asset_path, content_type


def create_http_server(host: str, port: int, runtime: QueryRuntime) -> ThreadingHTTPServer:
    """Create a local HTTP server bound to a supplied orchestrator-like runtime."""
    class PlatformHTTPServer(ThreadingHTTPServer):
        daemon_threads = True

    class Handler(PlatformHTTPRequestHandler):
        pass

    Handler.runtime = runtime
    return PlatformHTTPServer((host, port), Handler)


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
