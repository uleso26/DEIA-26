from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP
from core.runtime_utils import utc_now_iso


class BaseMCPStyleServer:
    """Thin wrapper around the MCP SDK while preserving local server ergonomics."""

    def __init__(self, server_name: str, instructions: str | None = None) -> None:
        self.server_name = server_name
        self._tools: dict[str, tuple[Callable[..., dict[str, Any]], str]] = {}
        self.app = FastMCP(
            name=server_name,
            instructions=instructions or f"{server_name} tool server",
            log_level="ERROR",
        )

    def register_tool(self, name: str, description: str, handler: Callable[..., dict[str, Any]]) -> None:
        self._tools[name] = (handler, description)
        self.app.add_tool(handler, name=name, description=description, structured_output=True)

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

    def serve(self, transport: str = "stdio") -> None:
        self.app.run(transport=transport)

    def run_cli(self) -> None:
        parser = argparse.ArgumentParser(description=f"{self.server_name} MCP server")
        parser.add_argument("--tool", help="Execute a single tool call and exit.")
        parser.add_argument("--arguments", default="{}", help="JSON arguments for one-shot execution.")
        parser.add_argument(
            "--transport",
            default="stdio",
            choices=["stdio", "sse", "streamable-http"],
            help="MCP transport to use when serving.",
        )
        args = parser.parse_args()
        if args.tool:
            payload = self.call_tool(args.tool, json.loads(args.arguments))
            print(json.dumps(payload, indent=2))
            return
        self.serve(transport=args.transport)
