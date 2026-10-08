
# -*- coding: utf-8 -*-
"""TCP Socket RAG service using the same RAGEngine as HTTP."""

import socket
import json
import struct
import threading
import signal

from server.runtime.service_runtime import create_service
from observability import trace_rag

from utils.config_loader import settings

HOST=settings.get("service.server.socket_host", "0.0.0.0")
PORT=int(settings.get("service.server.socket_port", 1234))

engine = create_service()


@trace_rag
def process(req):
    if req.get("cmd") == "health":
        return engine.health()
    if req.get("cmd") == "query":
        return engine.chat(req["text"])
    return {"error": "unknown command"}


_FRAME_HEADER = struct.Struct(">I")
_MAX_MSG = 64 * 1024 * 1024


def _recv_exact(conn, n):
    """精确读取 n 字节；TCP 是字节流，单次 recv 可能只返回部分数据。"""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接在消息读毕前关闭")
        buf += chunk
    return buf


def _recv_message(conn):
    (length,) = _FRAME_HEADER.unpack(_recv_exact(conn, _FRAME_HEADER.size))
    if length <= 0 or length > _MAX_MSG:
        raise ValueError(f"非法消息长度: {length}")
    return json.loads(_recv_exact(conn, length).decode("utf-8"))


def _send_message(conn, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    conn.sendall(_FRAME_HEADER.pack(len(payload)) + payload)


def client_handler(conn):
    with conn:
        while True:
            try:
                req = _recv_message(conn)
            except (ConnectionError, ValueError):
                break
            try:
                resp = process(req)
            except Exception as e:
                resp = {"error": str(e)}
            try:
                _send_message(conn, resp)
            except Exception:
                break


def start():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(20)
    # 让 accept() 周期性超时：否则主线程一直阻塞在 accept()，
    # Windows 下 Ctrl+C 的信号处理器要等 accept 返回才执行，表现为“只能 kill”。
    s.settimeout(1.0)
    print(f"RAG server {HOST}:{PORT}")

    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False
        try:
            s.close()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    try:
        while running:
            try:
                c, _ = s.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=client_handler, args=(c,), daemon=True).start()
    finally:
        try:
            s.close()
        except Exception:
            pass
    print("RAG server stopped.")


if __name__ == "__main__":
    start()
