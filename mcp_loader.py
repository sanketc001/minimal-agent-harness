"""Load every server listed in mcp.json and expose their tools.

This is the Model Context Protocol client in about 100 lines. Each server
is a subprocess we talk to over stdio using JSON-RPC 2.0. The handshake
is: initialize -> notifications/initialized -> tools/list. After that,
every tool call becomes a `tools/call` request.

The loader returns a list of tool specs ready to hand to OpenAI:
    [{"name": "mcp__fs__read_file", "schema": {...openai tool schema...},
      "fn": <closure that forwards the call to the right server>}]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any


class MCPServer:
    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str]):
        self.name = name
        full_env = os.environ.copy()
        full_env.update(env or {})
        # On Windows, bare names like `npx` need resolving to `npx.cmd`.
        resolved = shutil.which(command) or command
        self.proc = subprocess.Popen(
            [resolved, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=full_env,
            text=True,
            bufsize=1,
            encoding="utf-8",
        )
        self._id = 0
        self._lock = threading.Lock()

    def _send(self, payload: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _read_until_id(self, target_id: int, timeout: float = 30.0) -> dict:
        assert self.proc.stdout is not None
        # Line-delimited JSON-RPC. Skip notifications (no id) until our id arrives.
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server '{self.name}' closed stdout")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == target_id:
                return msg
            # else: notification/event, ignore

    def call(self, method: str, params: dict | None = None) -> Any:
        with self._lock:
            self._id += 1
            mid = self._id
            self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})
            resp = self._read_until_id(mid)
        if "error" in resp:
            raise RuntimeError(f"{self.name}: {resp['error']}")
        return resp.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def handshake(self) -> None:
        self.call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ccharness", "version": "0.1"},
            },
        )
        self.notify("notifications/initialized")

    def list_tools(self) -> list[dict]:
        return self.call("tools/list").get("tools", [])

    def stop(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass


def load(config_path: str | Path = "mcp.json") -> tuple[list[dict], list[MCPServer]]:
    """Start every server in mcp.json and return (tool_specs, servers)."""
    path = Path(config_path)
    if not path.exists():
        return [], []
    config = json.loads(path.read_text(encoding="utf-8"))
    servers: list[MCPServer] = []
    tools: list[dict] = []

    for name, cfg in (config.get("mcpServers") or {}).items():
        try:
            server = MCPServer(
                name=name,
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
            )
            server.handshake()
            for tool in server.list_tools():
                tool_name = f"mcp__{name}__{tool['name']}"
                tools.append({
                    "name": tool_name,
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool.get("description", ""),
                            "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                        },
                    },
                    "fn": _make_caller(server, tool["name"]),
                })
            servers.append(server)
            print(f"[mcp] {name}: {len(server.list_tools())} tools loaded")
        except Exception as e:
            print(f"[mcp] failed to start '{name}': {e}")

    return tools, servers


def _make_caller(server: MCPServer, tool_name: str):
    def _call(**kwargs) -> str:
        result = server.call("tools/call", {"name": tool_name, "arguments": kwargs})
        # MCP tool results are {"content": [{"type":"text","text":...}, ...]}
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts) if parts else json.dumps(result)
    return _call
