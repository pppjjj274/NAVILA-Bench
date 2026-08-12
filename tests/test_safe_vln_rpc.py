import pytest

import safe_vln.rpc as rpc
from safe_vln.rpc import (
    MAX_RPC_MESSAGE_BYTES,
    RemoteVLMError,
    bind_server_socket,
    error_payload,
    raise_for_remote_error,
    recv_json,
    send_json,
)


class MemoryConnection:
    def __init__(self, incoming=b"", chunk_size=3):
        self.incoming = bytearray(incoming)
        self.chunk_size = chunk_size
        self.sent = bytearray()

    def recv(self, size):
        count = min(size, self.chunk_size, len(self.incoming))
        if count == 0:
            return b""
        result = bytes(self.incoming[:count])
        del self.incoming[:count]
        return result

    def sendall(self, payload):
        self.sent.extend(payload)


class RecordingServerSocket:
    def __init__(self):
        self.calls = []

    def setsockopt(self, *args):
        self.calls.append(("setsockopt", args))

    def bind(self, address):
        self.calls.append(("bind", address))

    def listen(self, backlog):
        self.calls.append(("listen", backlog))

    def close(self):
        self.calls.append(("close",))


def test_server_socket_enables_address_reuse_before_bind(monkeypatch):
    server_socket = RecordingServerSocket()
    monkeypatch.setattr(rpc.socket, "socket", lambda *_args: server_socket)

    assert bind_server_socket("127.0.0.1", 54621) is server_socket
    assert server_socket.calls == [
        ("setsockopt", (rpc.socket.SOL_SOCKET, rpc.socket.SO_REUSEADDR, 1)),
        ("bind", ("127.0.0.1", 54621)),
        ("listen", 1),
    ]


def test_rpc_round_trip_handles_socket_framing():
    payload = {"images": ["abc"], "query": "go to the door"}
    sender = MemoryConnection()
    send_json(sender, payload)
    assert recv_json(MemoryConnection(sender.sent, chunk_size=2)) == payload


def test_rpc_rejects_truncated_payload():
    receiver = MemoryConnection((10).to_bytes(8, "big") + b"{}")
    with pytest.raises(ConnectionError, match="2 of 10 bytes"):
        recv_json(receiver)


def test_rpc_rejects_oversized_length_before_reading_payload():
    receiver = MemoryConnection(
        (MAX_RPC_MESSAGE_BYTES + 1).to_bytes(8, "big")
    )
    with pytest.raises(ValueError, match="invalid Safe-VLN RPC payload"):
        recv_json(receiver)


def test_rpc_rejects_non_finite_json_constants():
    payload = b'{"cost": NaN}'
    receiver = MemoryConnection(len(payload).to_bytes(8, "big") + payload)
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        recv_json(receiver)


def test_remote_server_error_is_explicit():
    payload = error_payload(ValueError("bad image"))
    with pytest.raises(RemoteVLMError, match="ValueError: bad image"):
        raise_for_remote_error(payload)
