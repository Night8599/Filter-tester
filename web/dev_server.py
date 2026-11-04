"""Run the web interface locally with regular Python (no Pyodide needed).

    pip install -e .
    python web/dev_server.py            # then open http://localhost:8000/?backend=server

Serves web/ and answers the page's API calls with filter_tester.web_api, so
changes to the Python package are picked up after a server restart.
Without ?backend=server the page loads Pyodide from the internet instead,
exactly as on GitHub Pages (that requires `python web/build.py` and serving
web/dist/).
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent / "src"))

from filter_tester import web_api  # noqa: E402

ALLOWED = {"scenarios", "filter_defaults", "create_session", "advance", "close_session"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_POST(self):
        if self.path.split("?")[0] != "/api":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            fn = body["fn"]
            if fn not in ALLOWED:
                raise ValueError(f"unknown function {fn!r}")
            result = getattr(web_api, fn)(*body.get("args", []))
            payload = {"result": json.loads(result) if isinstance(result, str) else result}
            status = 200
        except Exception as err:  # report errors to the page
            payload, status = {"error": f"{type(err).__name__}: {err}"}, 400
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if not (args and "/api" in str(args[0])):  # keep the console readable
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    port = parser.parse_args().port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Filter Tester UI: http://localhost:{port}/?backend=server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
