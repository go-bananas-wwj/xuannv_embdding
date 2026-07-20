"""Serve the shareable weekly briefing pages from the repository root."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path("/root/workspace/xuannv").resolve()
REPORT_ROOT = REPO_ROOT / "docs/reports/weekly_20260719"


class WeeklyReportHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        clean_path = unquote(urlparse(path).path)
        if clean_path in {"/", "/index.html"}:
            return str(REPORT_ROOT / "index.html")
        routes = {
            "/weekly/model-training/": "model_training_report.html",
            "/weekly/paper-progress/": "paper_progress_report.html",
            "/weekly/production-report/": "production_capability_report.html",
            # Keep links copied from the first report revision working.
            "/model_training_report.html": "model_training_report.html",
            "/paper_progress_report.html": "paper_progress_report.html",
            "/production_capability_report.html": "production_capability_report.html",
        }
        if clean_path in routes:
            return str(REPORT_ROOT / routes[clean_path])
        if clean_path.startswith("/weekly/"):
            return str(REPORT_ROOT / clean_path.removeprefix("/weekly/"))
        return str(REPO_ROOT / clean_path.lstrip("/"))

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), WeeklyReportHandler)
    print(f"Serving weekly reports on http://0.0.0.0:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
