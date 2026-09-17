"""HTTP range access to large remote files.

Both datasets in this project live as multi-gigabyte files on public servers, and both times we
want a small piece: the first 500 seconds of an IBL compressed binary, or a few HDF5 chunks out
of an Allen NWB file. Downloading the whole file to read 5% of it wastes hours, so this module
provides the two access patterns that avoid it.

:class:`HttpRangeFile` presents a remote URL as a seekable binary file object, fetching fixed
size blocks on demand and caching them. Handed to ``h5py.File``, it makes HDF5's own chunked
layout do the work: reading one channel's samples touches only the blocks holding that channel's
chunks. :func:`download` is the opposite choice, a whole file fetched once with resume support,
for when the access pattern is scattered enough that ranges lose.
"""

from __future__ import annotations

import io
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlsplit

#: Block size for :class:`HttpRangeFile`. HDF5 reads are small and scattered, so a block much
#: smaller than this multiplies request count, while a much larger one wastes bandwidth on a
#: metadata read of a few hundred bytes.
DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024

#: How many blocks to keep. HDF5 revisits its B-tree and heap blocks constantly, so even a small
#: cache turns most metadata reads into hits.
DEFAULT_MAX_BLOCKS = 64

_USER_AGENT = "lfpaudit/0.1 (+https://github.com/happyc0der/lfp2vec-audit)"


class RemoteError(RuntimeError):
    """A remote read failed after exhausting retries."""


def _request(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    request = urllib.request.Request(url, method=method)
    request.add_header("User-Agent", _USER_AGENT)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    return request


def _retrying(
    url: str,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    attempts: int = 4,
    timeout: float = 60.0,
):
    """Open a URL, retrying transient failures with exponential backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(_request(url, headers, method), timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            # A 4xx other than 429 will not improve by waiting, so fail immediately.
            status = getattr(error, "code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise RemoteError(f"{method} {url} failed with HTTP {status}") from error
            last = error
            if attempt < attempts - 1:
                time.sleep(2.0**attempt)
    raise RemoteError(f"{method} {url} failed after {attempts} attempts: {last!r}") from last


def content_length(url: str) -> int:
    """Size of a remote file in bytes, via a HEAD request."""
    with _retrying(url, method="HEAD") as response:
        length = response.headers.get("Content-Length")
    if length is None:
        raise RemoteError(f"server did not report a Content-Length for {url}")
    return int(length)


def supports_ranges(url: str) -> bool:
    """Whether the server advertises byte-range support."""
    with _retrying(url, method="HEAD") as response:
        return response.headers.get("Accept-Ranges", "").lower() == "bytes"


def read_range(url: str, start: int, stop: int) -> bytes:
    """Fetch bytes ``[start, stop)``. ``stop`` is exclusive, matching Python slicing."""
    if stop <= start:
        return b""
    # HTTP ranges are inclusive on both ends.
    headers = {"Range": f"bytes={start}-{stop - 1}"}
    with _retrying(url, headers=headers) as response:
        if response.status not in (200, 206):
            raise RemoteError(f"unexpected status {response.status} for a range request on {url}")
        payload = response.read()
    if response.status == 200 and len(payload) > stop - start:
        # Server ignored the range and sent everything; slice out what was asked for.
        payload = payload[start:stop]
    return payload


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file object backed by HTTP range requests.

    Reads are served from a small LRU cache of fixed-size blocks, so the scattered small reads
    HDF5 makes while walking its metadata mostly hit the cache while bulk data reads stream
    through it. ``h5py.File(HttpRangeFile(url), "r")`` then works against a remote NWB file.

    Close it explicitly, or use it as a context manager: h5py holding a Python file object can
    segfault if the interpreter tears down in the wrong order.
    """

    def __init__(
        self,
        url: str,
        size: int | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
    ) -> None:
        super().__init__()
        self.url = url
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self._size = int(size) if size is not None else content_length(url)
        self._position = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        #: Request and byte counters, reported by the loaders so cost is visible in the notebook.
        self.requests = 0
        self.bytes_fetched = 0

    # -- file protocol ------------------------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._position + offset
        elif whence == os.SEEK_END:
            target = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if target < 0:
            raise OSError("negative seek position")
        self._position = target
        return self._position

    @property
    def size(self) -> int:
        return self._size

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            self._blocks.move_to_end(index)
            return cached
        start = index * self.block_size
        stop = min(start + self.block_size, self._size)
        payload = read_range(self.url, start, stop)
        self.requests += 1
        self.bytes_fetched += len(payload)
        self._blocks[index] = payload
        while len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        return payload

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._position
        stop = min(self._position + size, self._size)

        pieces: list[bytes] = []
        position = self._position
        while position < stop:
            index = position // self.block_size
            block = self._block(index)
            offset = position - index * self.block_size
            take = min(len(block) - offset, stop - position)
            pieces.append(block[offset : offset + take])
            position += take
        self._position = position
        return b"".join(pieces)

    def readinto(self, buffer) -> int:  # noqa: ANN001 - buffer protocol
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)

    def close(self) -> None:
        self._blocks.clear()
        super().close()


def download(
    url: str,
    path: str | Path,
    expected_bytes: int | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
    progress: bool = True,
) -> Path:
    """Fetch a file, or a prefix of one, resuming a partial download if present.

    ``expected_bytes`` is both the completion target and, when it is smaller than the remote
    file, an explicit request for a prefix. That distinction matters: an earlier version treated
    it only as a target and sent no range header on a fresh download, so asking for the first few
    megabytes of an IBL recording quietly pulled all 3.2 GB and checked the size afterwards.
    Every request here now carries an explicit range whenever the caller wants less than
    everything.

    Returns the path. A file already at ``expected_bytes`` is left alone, so re-running a build
    command costs one HEAD request rather than a re-download.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    remote_total = content_length(url)
    total = remote_total if expected_bytes is None else int(expected_bytes)
    if total > remote_total:
        raise RemoteError(f"asked for {total} bytes but {url} holds only {remote_total}")

    have = path.stat().st_size if path.exists() else 0
    if have == total:
        return path
    if have > total:
        # A stale or longer partial file; start over rather than guess what it holds.
        path.unlink()
        have = 0

    headers = {}
    if have or total < remote_total:
        # Inclusive on both ends, unlike Python slicing.
        headers["Range"] = f"bytes={have}-{total - 1}"

    with _retrying(url, headers=headers) as response, open(path, "ab" if have else "wb") as handle:
        if headers and response.status != 206:
            raise RemoteError(
                f"{url} ignored a range request (status {response.status}); "
                "refusing to download the whole file"
            )
        written = have
        last_report = time.monotonic()
        while block := response.read(chunk_bytes):
            handle.write(block)
            written += len(block)
            if progress and time.monotonic() - last_report > 10:
                print(f"  {path.name}: {written / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                last_report = time.monotonic()

    final = path.stat().st_size
    if final != total:
        raise RemoteError(f"{path.name}: expected {total} bytes, got {final}")
    return path


def filename_from_url(url: str) -> str:
    """Last path segment of a URL, for naming a local cache file."""
    return Path(urlsplit(url).path).name
