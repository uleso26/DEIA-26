from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Callable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class BaseMCPStyleServer:
    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self._tools: dict[str, tuple[Callable[..., dict[str, Any]], str]] = {}

    def register_tool(self, name: str, description: str, handler: Callable[..., dict[str, Any]]) -> None:
        self._tools[name] = (handler, description)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": description}
            for name, (_, description) in sorted(self._tools.items())
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        handler, _ = self._tools[name]
        result = handler(**arguments)
        result.setdefault("requested_at", utc_now_iso())
        result.setdefault("source_metadata", {})
        result["server_name"] = self.server_name
        return result

    def serve(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                action = request.get("action")
                if action == "list_tools":
                    response = {"ok": True, "tools": self.list_tools()}
                elif action == "call_tool":
                    response = {"ok": True, "result": self.call_tool(request["tool"], request.get("arguments", {}))}
                elif action == "shutdown":
                    response = {"ok": True}
                    print(json.dumps(response), flush=True)
                    break
                else:
                    response = {"ok": False, "error": f"Unknown action: {action}"}
            except Exception as exc:  # pragma: no cover - defensive I/O path
                response = {"ok": False, "error": str(exc)}
            print(json.dumps(response), flush=True)

    def run_cli(self) -> None:
        parser = argparse.ArgumentParser(description=f"{self.server_name} MCP-style server")
        parser.add_argument("--tool", help="Execute a single tool call and exit.")
        parser.add_argument("--arguments", default="{}", help="JSON arguments for one-shot execution.")
        args = parser.parse_args()
        if args.tool:
            payload = self.call_tool(args.tool, json.loads(args.arguments))
            print(json.dumps(payload, indent=2))
            return
        self.serve()
