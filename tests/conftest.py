"""Local HTTP server with Range support and in-memory archives/rasters.

Identical copy in meiazero/allclear, meiazero/sen12mscrts and meiazero/sen1floods11: change all.
"""

import io
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import rasterio

FILES: dict[str, bytes] = {}


class RangeHandler(BaseHTTPRequestHandler):
    honor_range = True

    def do_GET(self) -> None:
        body = FILES.get(self.path)
        if body is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and self.honor_range:
            start = int(rng.split("=")[1].rstrip("-"))
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def make_tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def make_tif(bands: int | np.ndarray, size: int = 256, dtype: str = "float64") -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.random((bands, size, size)) if isinstance(bands, int) else bands
    with rasterio.MemoryFile() as mem:
        with mem.open(
            driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
            dtype=dtype, transform=rasterio.transform.from_origin(0, 256, 1, 1),
        ) as d:  # fmt: skip
            d.write(arr.astype(dtype))
        return mem.read()
