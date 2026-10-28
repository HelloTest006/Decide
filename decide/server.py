"""
Lightweight stdlib + TypeSafe Jev / Laya wire-compatible HTTP Server for DecideRuntime.
Serves `POST /v1/systemone`, `POST /v1/systemone/batch`, `GET /v1/models`, and `GET /health`
without requiring external web framework dependencies.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from .runtime import DecideRuntime


def serve_runtime(runtime: Optional[DecideRuntime] = None, host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    rt = runtime or DecideRuntime()

    class SystemOneHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, data: dict):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "models": ["decide-v1", "decidex-v1"]})
            elif self.path in ("/v1/models", "/models"):
                self._send_json(
                    200,
                    {
                        "data": [
                            {"id": "decide-v1", "family": "Decide", "aliases": ["decide", "standard", "fast"]},
                            {"id": "decidex-v1", "family": "DecideX", "aliases": ["decidex", "large", "x"]},
                        ]
                    },
                )
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path in ("/v1/systemone", "/predict"):
                    res = rt.handle_systemone_request(payload)
                    self._send_json(200, res)
                elif self.path in ("/v1/systemone/batch", "/predict/batch"):
                    states = payload.get("states", [])
                    questions = payload.get("questions", {})
                    model = payload.get("model", "auto")
                    min_conf = payload.get("min_confidence")
                    results = rt.predict_batch(states, questions, model=model, min_confidence=min_conf)
                    self._send_json(200, {"results": results})
                else:
                    self._send_json(404, {"error": "not found"})
            except Exception as exc:
                self._send_json(422, {"error": str(exc)})

        def log_message(self, format, *args):
            return

    return HTTPServer((host, port), SystemOneHandler)
