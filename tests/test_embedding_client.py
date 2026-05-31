from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent_recall_store import EmbeddingClient


class Handler(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        Handler.received.append({"path": self.path, "payload": payload, "auth": self.headers.get("Authorization")})
        body = json.dumps({"data": [{"embedding": [0.1, 0.2, 0.3]}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def test_embedding_client_calls_only_embeddings_endpoint_and_omits_dimensions_when_zero():
    Handler.received.clear()
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = EmbeddingClient(f"http://127.0.0.1:{server.server_port}/v1", "embed-model", api_key="k", dimensions=0)
        assert client.embed("hello") == [0.1, 0.2, 0.3]
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert Handler.received[0]["path"] == "/v1/embeddings"
    assert Handler.received[0]["payload"] == {"model": "embed-model", "input": "hello"}
    assert Handler.received[0]["auth"] == "Bearer k"


def test_embedding_client_sends_dimensions_when_positive():
    Handler.received.clear()
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = EmbeddingClient(f"http://127.0.0.1:{server.server_port}/v1", "embed-model", dimensions=2560)
        client.embed("hello")
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert Handler.received[0]["payload"]["dimensions"] == 2560
