from __future__ import annotations

import atexit
import json
import subprocess
from typing import Any

from core.paths import ROOT


SERVER_MODULES = {
    "safety": "mcp_servers.safety_server",
    "trials": "mcp_servers.trials_server",
    "knowledge": "mcp_servers.knowledge_server",
}


class StdioMCPConnection:
    def __init__(self, module_name: str) -> None:
        self.process = subprocess.Popen(
            ["python3", "-m", module_name],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("MCP connection was not initialized correctly.")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"MCP server did not respond. stderr={stderr.strip()}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown MCP error"))
        return response

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request({"action": "list_tools"})["tools"]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request({"action": "call_tool", "tool": tool_name, "arguments": arguments})["result"]

    def close(self) -> None:
        if self.process.poll() is not None:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()
            return
        try:
            self.request({"action": "shutdown"})
        except Exception:
            self.process.terminate()
        finally:
            try:
                self.process.wait(timeout=5)
            finally:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()


class MCPClientManager:
    def __init__(self) -> None:
        self._connections: dict[str, StdioMCPConnection] = {}
        atexit.register(self.close_all)

    def _get_connection(self, server_name: str) -> StdioMCPConnection:
        if server_name not in SERVER_MODULES:
            raise KeyError(f"Unknown server: {server_name}")
        if server_name not in self._connections:
            self._connections[server_name] = StdioMCPConnection(SERVER_MODULES[server_name])
        return self._connections[server_name]

    def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        return self._get_connection(server_name).list_tools()

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._get_connection(server_name).call_tool(tool_name, arguments)

    def close_all(self) -> None:
        for connection in list(self._connections.values()):
            try:
                connection.close()
            except Exception:
                pass
        self._connections.clear()
