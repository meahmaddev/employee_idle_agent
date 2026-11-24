"""Live annotated view in the browser (MJPEG over HTTP).

The pipeline pushes each annotated frame here; open http://localhost:<port>
to watch it. The server binds to 127.0.0.1 only, so on a rented GPU reach it
through an SSH tunnel (ssh ... -L 8090:localhost:8090) instead of exposing it.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Idle agent - live view</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font-family: sans-serif; }
  header { padding: 8px 12px; font-size: 14px; }
  img { display: block; max-width: 100vw; max-height: calc(100vh - 40px); margin: 0 auto; }
</style></head>
<body><header>Employee idle agent &mdash; live view (boxes, zones and states as the model sees them)</header>
<img src="/stream" alt="Waiting for frames..."></body></html>
"""


class LiveView:
    """Holds the latest annotated frame and serves it as an MJPEG stream."""

    def __init__(self, port: int, host: str = "127.0.0.1", scale: float = 0.75, quality: int = 70):
        self.port = port
        self.host = host
        self.scale = scale
        self.quality = quality
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stopped = False
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ server
    def start(self) -> None:
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # keep the pipeline log clean
                return

            def do_GET(self):  # noqa: N802 (http.server naming)
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", _PAGE)
                elif path == "/frame.jpg":
                    with view._cond:
                        data = view._jpeg
                    if data is None:
                        self._send(503, "text/plain", b"No frame yet")
                    else:
                        self._send(200, "image/jpeg", data)
                elif path == "/stream":
                    self._stream()
                else:
                    self._send(404, "text/plain", b"Not found")

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_seq = -1
                try:
                    while True:
                        with view._cond:
                            view._cond.wait_for(
                                lambda: view._stopped or view._seq != last_seq, timeout=5.0
                            )
                            if view._stopped:
                                return
                            if view._seq == last_seq or view._jpeg is None:
                                continue
                            data, last_seq = view._jpeg, view._seq
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(data)).encode()
                            + b"\r\n\r\n"
                            + data
                            + b"\r\n"
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return  # browser tab closed

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Live view → http://localhost:{self.port}  (on a remote GPU: ssh ... -L {self.port}:localhost:{self.port})")

    # ----------------------------------------------------------------- frames
    def push(self, frame: np.ndarray) -> None:
        """Encode and publish one annotated BGR frame."""
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        time.sleep(0.05)
