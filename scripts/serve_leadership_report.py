from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path("/root/workspace/xuannv").resolve()
DATA_ROOT = Path("/data/xuannv_embedding").resolve()
REPORT = REPO_ROOT / "docs/reports/leadership_embedding_report_20260629.html"


class ReportHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean_path = unquote(parsed.path)
        if clean_path in {"/", "/index.html", "/report.html"}:
            return str(REPORT)
        if clean_path.startswith("/data/xuannv_embedding/"):
            relative = clean_path.removeprefix("/data/xuannv_embedding/")
            return str(DATA_ROOT / relative)
        relative = clean_path.lstrip("/")
        return str(REPO_ROOT / relative)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8001), ReportHandler)
    print("Serving leadership report on http://0.0.0.0:8001/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
