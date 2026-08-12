"""Small fail-closed JSON protocol shared by the VLM client and server."""

from __future__ import annotations

import json
import socket
from typing import Any


RPC_PROTOCOL_VERSION = "safe-vln-rpc-v1"
MAX_RPC_MESSAGE_BYTES = 256 * 1024 * 1024


class RemoteVLMError(RuntimeError):
    """An explicit error returned by the VLM server."""


def bind_server_socket(host: str, port: int, backlog: int = 1) -> socket.socket:
    """Bind a restart-safe TCP listener for the local VLM RPC service."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(backlog)
    except Exception:
        server_socket.close()
        raise
    return server_socket


def recv_exact(connection: socket.socket, size: int) -> bytes:
    if size < 0:
        raise ValueError("socket payload size must be non-negative")
    chunks = bytearray()
    while len(chunks) < size:
        packet = connection.recv(size - len(chunks))
        if not packet:
            raise ConnectionError(
                f"socket closed after {len(chunks)} of {size} bytes"
            )
        chunks.extend(packet)
    return bytes(chunks)


def send_json(connection: socket.socket, payload: Any) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )
    if len(encoded) > MAX_RPC_MESSAGE_BYTES:
        raise ValueError("Safe-VLN RPC payload is too large")
    connection.sendall(len(encoded).to_bytes(8, "big"))
    connection.sendall(encoded)


def recv_json(connection: socket.socket) -> Any:
    size = int.from_bytes(recv_exact(connection, 8), "big")
    if size <= 0 or size > MAX_RPC_MESSAGE_BYTES:
        raise ValueError(f"invalid Safe-VLN RPC payload length: {size}")
    try:
        return json.loads(
            recv_exact(connection, size).decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Safe-VLN RPC payload is not valid UTF-8 JSON") from error


def error_payload(error: Exception) -> dict[str, Any]:
    return {
        "rpc_protocol_version": RPC_PROTOCOL_VERSION,
        "ok": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def raise_for_remote_error(payload: Any) -> Any:
    if not isinstance(payload, dict) or payload.get("ok") is not False:
        return payload
    error = payload.get("error")
    if isinstance(error, dict):
        error_type = str(error.get("type") or "RemoteError")
        message = str(error.get("message") or "VLM request failed")
    else:
        error_type = "RemoteError"
        message = "VLM request failed"
    raise RemoteVLMError(f"VLM server {error_type}: {message}")


__all__ = [
    "MAX_RPC_MESSAGE_BYTES",
    "RPC_PROTOCOL_VERSION",
    "RemoteVLMError",
    "bind_server_socket",
    "error_payload",
    "raise_for_remote_error",
    "recv_exact",
    "recv_json",
    "send_json",
]
