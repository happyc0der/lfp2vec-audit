"""Tests for range-based remote access, served by a local HTTP server.

No test here touches the network: a stdlib server on a loopback port serves a temporary file,
which exercises the same code paths as a real S3 object including range handling.
"""

from __future__ import annotations

import http.server
import threading
from functools import partial

import pytest

from lfpaudit.data.remote import (
    HttpRangeFile,
    RemoteError,
    content_length,
    download,
    filename_from_url,
    read_range,
)

PAYLOAD = bytes(range(256)) * 400  # 102 400 bytes, every byte position identifiable


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves one fixed payload, honouring Range requests like a real object store."""

    payload = PAYLOAD
    honour_ranges = True

    def log_message(self, *args: object) -> None:  # keep test output clean
        pass

    def _send(self, body: bytes, status: int, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes" if self.honour_ranges else "none")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self._send(self.payload, 200)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        header = self.headers.get("Range")
        if not header or not self.honour_ranges:
            self._send(self.payload, 200)
            return
        spec = header.split("=")[1]
        start_text, _, stop_text = spec.partition("-")
        start = int(start_text)
        stop = int(stop_text) + 1 if stop_text else len(self.payload)
        body = self.payload[start:stop]
        self._send(body, 206, {"Content-Range": f"bytes {start}-{stop - 1}/{len(self.payload)}"})


@pytest.fixture
def server():
    """A loopback HTTP server serving :data:`PAYLOAD`; yields its base URL."""

    def _make(honour_ranges: bool = True):
        handler = type("Handler", (_RangeHandler,), {"honour_ranges": honour_ranges})
        httpd = http.server.HTTPServer(("127.0.0.1", 0), partial(handler))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/blob.bin"

    created = []

    def factory(honour_ranges: bool = True) -> str:
        httpd, url = _make(honour_ranges)
        created.append(httpd)
        return url

    yield factory
    for httpd in created:
        httpd.shutdown()
        httpd.server_close()


def test_content_length(server):
    assert content_length(server()) == len(PAYLOAD)


def test_read_range_returns_exact_bytes(server):
    url = server()
    assert read_range(url, 10, 20) == PAYLOAD[10:20]
    assert read_range(url, 0, 1) == PAYLOAD[0:1]
    assert read_range(url, len(PAYLOAD) - 5, len(PAYLOAD)) == PAYLOAD[-5:]
    assert read_range(url, 30, 30) == b""


def test_range_file_reads_like_a_file(server):
    with HttpRangeFile(server(), block_size=1024) as handle:
        assert handle.read(10) == PAYLOAD[:10]
        assert handle.tell() == 10
        handle.seek(5000)
        assert handle.read(100) == PAYLOAD[5000:5100]
        handle.seek(-20, 2)
        assert handle.read() == PAYLOAD[-20:]
        assert handle.read() == b""


def test_range_file_spans_block_boundaries(server):
    with HttpRangeFile(server(), block_size=1024) as handle:
        handle.seek(1000)
        # Straddles three blocks, so the result is only correct if stitching works.
        assert handle.read(2100) == PAYLOAD[1000:3100]


def test_range_file_caches_blocks(server):
    with HttpRangeFile(server(), block_size=1024) as handle:
        handle.seek(0)
        handle.read(100)
        first = handle.requests
        handle.seek(0)
        handle.read(100)
        assert handle.requests == first, "a repeated read should hit the cache"


def test_range_file_evicts_old_blocks(server):
    with HttpRangeFile(server(), block_size=1024, max_blocks=2) as handle:
        for offset in (0, 2048, 4096):
            handle.seek(offset)
            handle.read(10)
        before = handle.requests
        handle.seek(0)  # evicted by now
        handle.read(10)
        assert handle.requests == before + 1


def test_range_file_is_seekable_and_read_only(server):
    with HttpRangeFile(server()) as handle:
        assert handle.readable() and handle.seekable()
        assert not handle.writable()
        assert handle.size == len(PAYLOAD)


def test_download_whole_file(tmp_path, server):
    path = download(server(), tmp_path / "a.bin", progress=False)
    assert path.read_bytes() == PAYLOAD


def test_download_prefix_requests_only_that_range(tmp_path, server):
    path = download(server(), tmp_path / "p.bin", expected_bytes=500, progress=False)
    assert path.read_bytes() == PAYLOAD[:500]


def test_download_is_idempotent(tmp_path, server):
    url = server()
    path = download(url, tmp_path / "p.bin", expected_bytes=500, progress=False)
    mtime = path.stat().st_mtime_ns
    again = download(url, tmp_path / "p.bin", expected_bytes=500, progress=False)
    assert again.stat().st_mtime_ns == mtime, "an already complete file should not be refetched"


def test_download_resumes_a_partial_file(tmp_path, server):
    path = tmp_path / "p.bin"
    path.write_bytes(PAYLOAD[:200])
    download(server(), path, expected_bytes=500, progress=False)
    assert path.read_bytes() == PAYLOAD[:500]


def test_download_restarts_when_local_file_is_too_long(tmp_path, server):
    path = tmp_path / "p.bin"
    path.write_bytes(PAYLOAD[:900])
    download(server(), path, expected_bytes=500, progress=False)
    assert path.read_bytes() == PAYLOAD[:500]


def test_download_refuses_when_server_ignores_ranges(tmp_path, server):
    """A server that ignores a prefix request must fail loudly, not send gigabytes."""
    url = server(honour_ranges=False)
    with pytest.raises(RemoteError, match="ignored a range request"):
        download(url, tmp_path / "p.bin", expected_bytes=500, progress=False)


def test_download_rejects_impossible_size(tmp_path, server):
    with pytest.raises(RemoteError, match="holds only"):
        download(server(), tmp_path / "p.bin", expected_bytes=len(PAYLOAD) + 1, progress=False)


def test_filename_from_url():
    assert filename_from_url("https://host/a/b/c.nwb?x=1") == "c.nwb"
