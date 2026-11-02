"""A recorded SealKeeper and a recorded MCP server, both served through
httpx.MockTransport, SEALs signed with a fixed test key, and a fake chat
model that calls one tool. No test reaches the network.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from sealkeeper import AsyncSealKeeper, SealKeeper

MCP_URL = "https://api.sealkeeper.run/mcp"
KEYS_URL = "https://sealkeeper.run/.well-known/seal.json"
# 1 October 2026 12:00 UTC. Every SEAL here is current at it.
NOW = 1790856000


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _public(key: Ed25519PrivateKey) -> str:
    return b64url(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


_SIGNER = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
KID = "test-1"
KEYS = {
    "keys": [
        {
            "kid": KID,
            "kty": "OKP",
            "crv": "Ed25519",
            "alg": "EdDSA",
            "x": _public(_SIGNER),
        }
    ]
}

HELPER_ID = _public(Ed25519PrivateKey.from_private_bytes(bytes([1] * 32)))
SERVICE_ID = _public(Ed25519PrivateKey.from_private_bytes(bytes([2] * 32)))
SERVER_URL = "https://mcp.example.com/mcp"
OTHER_URL = "https://other.example.com/mcp"


def sign(payload: dict[str, Any]) -> str:
    head = b64url(json.dumps({"alg": "EdDSA", "kid": KID}).encode())
    body = b64url(json.dumps(payload).encode())
    signature = b64url(_SIGNER.sign(f"{head}.{body}".encode()))
    return f"{head}.{body}.{signature}"


def seal_payload(sub: str, kind: str, **extra: Any) -> dict[str, Any]:
    return {
        "iss": "sealkeeper.run",
        "sub": sub,
        "ver": 9,
        "iat": NOW - 3600,
        "exp": NOW - 3600 + 86400,
        "subject": {"kind": kind},
        "level": "bronze",
        "trust": 412,
        "scores": {"reliability": 0.9},
        **extra,
    }


HELPER_SEAL = sign(seal_payload(HELPER_ID, "agent"))
SERVICE_SEAL = sign(
    seal_payload(
        SERVICE_ID,
        "service",
        listing={"handle": "example.com/mcp", "card_url": SERVER_URL},
    )
)
# A SEAL whose payload was changed after it was signed.
_head, _body, _signature = sign(seal_payload(HELPER_ID, "agent")).split(".")
BROKEN_SEAL = ".".join(
    [
        _head,
        b64url(json.dumps(seal_payload(HELPER_ID, "agent", level="gold")).encode()),
        _signature,
    ]
)


def shown(agent_id: str, handle: str, seal: str, **agent: Any) -> dict[str, Any]:
    return {
        "skill": "get-trust",
        "status": "shown",
        "agent": {"id": agent_id, "handle": handle, **agent},
        "seal": seal,
    }


NOT_FOUND = {"skill": "get-trust", "status": "not_found", "agent": None, "seal": None}

# What get_trust answers, by the target it is sent. Any other is not found.
ANSWERS: dict[str, dict[str, Any]] = {
    "alice/helper": shown(HELPER_ID, "alice/helper", HELPER_SEAL),
    # The unsigned agent fields claim gold. Only a verified SEAL is read.
    "alice/broken": shown(
        HELPER_ID, "alice/broken", BROKEN_SEAL, level="gold", trust=999
    ),
    SERVER_URL: shown(SERVICE_ID, "example.com/mcp", SERVICE_SEAL),
}


def tool_result(data: dict[str, Any]) -> dict[str, Any]:
    """A tools/call answer the way SealKeeper writes one."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {"type": "text", "text": "answer"},
                {"type": "text", "text": json.dumps(data)},
            ],
            "structuredContent": data,
        },
    }


@dataclass
class RecordedSealKeeper:
    """SealKeeper's MCP server and keys. mcp_status answers every get_trust
    with that status instead, and every request is kept."""

    mcp_status: int = 200
    requests: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url == KEYS_URL:
            return httpx.Response(200, json=KEYS)
        if url != MCP_URL:
            raise AssertionError(f"no reply for {request.method} {url}")
        if self.mcp_status != 200:
            return httpx.Response(self.mcp_status, json={"error": "unavailable"})
        target = json.loads(request.content)["params"]["arguments"]["agent"]
        return httpx.Response(200, json=tool_result(ANSWERS.get(target, NOT_FOUND)))

    def targets(self) -> list[str]:
        return [
            json.loads(r.content)["params"]["arguments"]["agent"]
            for r in self.requests
            if str(r.url) == MCP_URL
        ]


def clock() -> float:
    return NOW


def _no_sleep(_: float) -> None:
    return None


async def _no_asleep(_: float) -> None:
    return None


def sync_client(server: RecordedSealKeeper, **kw: Any) -> SealKeeper:
    http = httpx.Client(transport=httpx.MockTransport(server.handle))
    return SealKeeper(http_client=http, clock=clock, sleep=_no_sleep, **kw)


def async_client(server: RecordedSealKeeper, **kw: Any) -> AsyncSealKeeper:
    http = httpx.AsyncClient(transport=httpx.MockTransport(server.handle))
    return AsyncSealKeeper(http_client=http, clock=clock, sleep=_no_asleep, **kw)


class RecordedMcpServer:
    """An MCP server over streamable HTTP with one tool, tool_name, which
    echoes its text. It answers JSON, sends no session id and keeps the
    method of every message."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.methods: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)
        self.methods.append(message["method"])
        if "id" not in message:
            return httpx.Response(202)
        method = message["method"]
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "recorded", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": self.tool_name,
                        "description": "Echoes its text.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            text = message["params"]["arguments"]["text"]
            result = {"content": [{"type": "text", "text": f"echo {text}"}]}
        else:
            raise AssertionError(f"no reply for {method}")
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    def factory(self) -> Callable[..., httpx.AsyncClient]:
        """An httpx_client_factory for a connection to this server."""

        def make(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=httpx.MockTransport(self.handle),
                headers=headers,
                timeout=timeout,
                auth=auth,
            )

        return make


def http_connection(url: str, server: RecordedMcpServer) -> dict[str, Any]:
    return {
        "transport": "streamable_http",
        "url": url,
        "httpx_client_factory": server.factory(),
    }


class FakeToolCallingModel(FakeMessagesListChatModel):
    """A chat model that answers its responses in turn and takes any tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolCallingModel:
        return self


def calls_then_done(name: str, args: dict[str, Any]) -> FakeToolCallingModel:
    """A model that calls one tool, then answers done once it reads the
    tool's message."""
    return FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": "call-1"}],
            ),
            AIMessage(content="done"),
        ]
    )
