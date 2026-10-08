# -*- coding: utf-8 -*-
import socket
import json
import sys
import struct

HOST = "127.0.0.1"
PORT = 1234

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


def query(text):
    s = socket.socket()
    s.connect((HOST, PORT))
    body = json.dumps({"cmd": "query", "text": text}, ensure_ascii=False).encode()
    # 长度帧定界：4 字节大端长度头 + JSON 载荷。
    s.sendall(_FRAME_HEADER.pack(len(body)) + body)
    (length,) = _FRAME_HEADER.unpack(_recv_exact(s, _FRAME_HEADER.size))
    if length <= 0 or length > _MAX_MSG:
        s.close()
        raise ValueError(f"非法消息长度: {length}")
    data = json.loads(_recv_exact(s, length).decode())
    s.close()
    return data


if __name__ == "__main__":
    print(json.dumps(query(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))
