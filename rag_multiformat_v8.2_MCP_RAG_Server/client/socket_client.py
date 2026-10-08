# -*- coding: utf-8 -*-
"""TCP Socket client for Industrial RAG service."""
import socket
import json
import struct

_FRAME_HEADER = struct.Struct(">I")
_MAX_MSG = 64 * 1024 * 1024


def _recv_exact(sock, n):
    """精确读取 n 字节；TCP 是字节流，单次 recv 可能只返回部分数据。"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接在消息读毕前关闭")
        buf += chunk
    return buf


class RAGSocketClient:
    def __init__(self, host="127.0.0.1", port=1234, timeout=120):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _request(self, payload):
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            # 长度帧定界：4 字节大端长度头 + JSON 载荷，避免大响应被 TCP 分段
            # 导致单次 recv 拿到不完整 JSON 而解析失败。
            s.sendall(_FRAME_HEADER.pack(len(body)) + body)
            (length,) = _FRAME_HEADER.unpack(_recv_exact(s, _FRAME_HEADER.size))
            if length <= 0 or length > _MAX_MSG:
                raise ValueError(f"非法消息长度: {length}")
            return json.loads(_recv_exact(s, length).decode("utf-8"))

    def chat(self, question):
        return self._request({"cmd": "query", "text": question})

    def health(self):
        return self._request({"cmd": "health"})


if __name__ == "__main__":
    import sys
    c = RAGSocketClient()
    print(json.dumps(c.chat(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))
