#!/usr/bin/env python3
"""Reliable local HTTP Range gateway for remote COGs.

GDAL/libcurl occasionally receives truncated Azure Blob range responses on the
current cross-border route.  This gateway keeps GDAL on localhost and forwards
each range through ``requests`` with strict Content-Range validation and
bounded retries.  It never downloads a full COG unless the client asks for it.
"""

from __future__ import annotations

import argparse
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import requests


def _expected_length(range_header: str | None) -> int | None:
    if not range_header or not range_header.startswith("bytes="):
        return None
    value = range_header.removeprefix("bytes=").split(",", 1)[0]
    start, _, end = value.partition("-")
    if not start or not end:
        return None
    return int(end) - int(start) + 1


class RangeGateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout_seconds = 45
    attempts = 4

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _target(self) -> str:
        values = parse_qs(urlparse(self.path).query).get("url", [])
        if len(values) != 1 or not values[0].startswith("https://"):
            raise ValueError("expected exactly one HTTPS url query parameter")
        return values[0]

    def _fetch(self, target: str, range_header: str | None) -> requests.Response:
        headers = {"Range": range_header} if range_header else {}
        expected_length = _expected_length(range_header)
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                response = requests.get(target, headers=headers, timeout=(10, self.timeout_seconds), proxies={})
                if range_header:
                    if response.status_code != HTTPStatus.PARTIAL_CONTENT:
                        raise RuntimeError(f"upstream status {response.status_code}")
                    if expected_length is not None and len(response.content) != expected_length:
                        raise RuntimeError(f"truncated range {len(response.content)} != {expected_length}")
                    if not response.headers.get("Content-Range", "").startswith("bytes "):
                        raise RuntimeError("upstream did not return Content-Range")
                elif response.status_code >= 400:
                    response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        assert last_error is not None
        raise last_error

    def _serve(self, send_body: bool) -> None:
        try:
            target = self._target()
            requested_range = self.headers.get("Range")
            if not requested_range and not send_body:
                requested_range = "bytes=0-0"
            response = self._fetch(target, requested_range)
            status = response.status_code
            body = response.content if send_body else b""
            self.send_response(status)
            for name in ("Content-Range", "Content-Type", "Accept-Ranges", "ETag", "Last-Modified"):
                if value := response.headers.get(name):
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(body) if send_body else response.headers.get("Content-Length", "0")))
            self.end_headers()
            if send_body:
                self.wfile.write(body)
        except Exception as exc:
            payload = f"gateway error: {type(exc).__name__}: {exc}\n".encode()
            self.send_response(HTTPStatus.BAD_GATEWAY)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if send_body:
                self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._serve(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(send_body=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RangeGateway)
    print(f"COG range gateway listening at http://{args.host}:{args.port}/cog", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
