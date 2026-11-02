"""Serve the web demo locally (the ONNX model needs http, not file://).   python scripts/serve.py"""
import http.server
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
PORT = 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream"}


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), partial(Handler, directory=str(DOCS))) as srv:
        print(f"Demo at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        srv.serve_forever()
