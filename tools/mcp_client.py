from __future__ import annotations

import atexit
import json
import sys
from typing import Any

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from core.paths import ROOT
from core.runtime_utils import utc_now_iso


SERVER_MODULES = {
    "safety": "mcp_servers.safety_server",
    "trials": "mcp_servers.trials_server",
    "knowledge": "mcp_servers.knowledge_server",
}


class StdioMCPConnection:
    """Request-scoped MCP client over stdio using the official Python SDK."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def _server_parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", self.module_name],
            cwd=str(ROOT),
        )

    @staticmethod
    def _deserialize_tool_result(result: Any) -> dict[str, Any]:
        if getattr(result, "isError", False):
            fragments = []
            for content in getattr(result, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    fragments.append(text)
            raise RuntimeError(" ".join(fragments).strip() or "Unknown MCP tool error")
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured
        for content in getattr(result, "content", []) or []:
            text = getattr(content, "text", None)
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError("MCP tool call returned no structured payload.")

    async def _list_tools_async(self) -> list[dict[str, Any]]:
        async with stdio_client(self._server_parameters()) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return [tool.model_dump(by_alias=True, exclude_none=True) for tool in result.tools]

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with stdio_client(self._server_parameters()) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return self._deserialize_tool_result(result)

    def list_tools(self) -> list[dict[str, Any]]:
        return anyio.run(self._list_tools_async)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return anyio.run(self._call_tool_async, tool_name, arguments)

    def close(self) -> None:
        return
class MCPClientManager:
    """Lazily manage MCP stdio clients for the local tool servers."""

    def __init__(self) -> None:
        self._connections: dict[str, StdioMCPConnection] = {}
        atexit.register(self.close_all)

    def _get_connection(self, server_name: str) -> StdioMCPConnection:
        if server_name not in SERVER_MODULES:
            raise KeyError(f"Unknown server: {server_name}")
        if server_name not in self._connections:
            self._connections[server_name] = StdioMCPConnection(SERVER_MODULES[server_name])
        return self._connections[server_name]

    @staticmethod
    def _enrich_result(server_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched.setdefault("requested_at", utc_now_iso())
        enriched.setdefault("server_name", server_name)
        enriched.setdefault("source_metadata", {})
        return enriched

    def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        return self._get_connection(server_name).list_tools()

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self._get_connection(server_name).call_tool(tool_name, arguments)
        return self._enrich_result(server_name, payload)

    def close_all(self) -> None:
        for connection in list(self._connections.values()):
            try:
                connection.close()
            except Exception:
                pass
        self._connections.clear()
