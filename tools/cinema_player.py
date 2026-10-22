#!/usr/bin/env python3
"""
VIDEX Native Cinema Player
==========================

Minimal zero-dependency HTTP server that streams the rendered VIDEX films to a
browser with proper `206 Partial Content` byte-range support (needed for smooth
scrubbing of large MP4s).

    python3 tools/cinema_player.py --port 8080
"""

from __future__ import annotations

import argparse
import html
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VIDEX Native Cinema</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; background: #06080d;
    background-image: radial-gradient(ellipse at 50% -10%, #16243b 0%, #06080d 60%);
    color: #e8eef8; font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; align-items: center; padding: 28px 18px 60px;
  }}
  header {{ width: 100%; max-width: 1180px; display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
  h1 {{ font-size: 20px; letter-spacing: .12em; text-transform: uppercase; margin: 0; font-weight: 650; }}
  h1 span {{ color: #5ee0ff; }}
  .tag {{ font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: #7f8ea3; }}
  .badge {{ font-size: 11px; padding: 3px 9px; border-radius: 999px; border: 1px solid #1e3a52;
            color: #7ff0c4; background: #0c211b; letter-spacing: .08em; }}
  .stage {{ width: 100%; max-width: 1180px; margin-top: 18px; border-radius: 14px; overflow: hidden;
            background: #000; border: 1px solid #17222f; box-shadow: 0 30px 80px -30px #000; }}
  video {{ display: block; width: 100%; aspect-ratio: {ar}; background: #000; }}
  .bar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 16px; width: 100%; max-width: 1180px; }}
  .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  a.tab, a.dl {{ text-decoration: none; font-size: 13px; padding: 8px 14px; border-radius: 9px;
             border: 1px solid #1d2b3a; color: #b9c8dc; background: #0b1119; }}
  a.tab.active {{ border-color: #2d6f8a; color: #9fe9ff; background: #0d1c26; }}
  a.dl {{ margin-left: auto; border-color: #2b5c3f; color: #9df0bd; background: #0a1a13; }}
  .meta {{ width: 100%; max-width: 1180px; margin-top: 22px; display: grid;
           grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
  .cell {{ border: 1px solid #16222e; border-radius: 10px; padding: 11px 13px; background: #080d13; }}
  .k {{ font-size: 10px; letter-spacing: .16em; text-transform: uppercase; color: #6d7f95; }}
  .v {{ font-size: 14px; color: #dbe7f6; margin-top: 3px; }}
</style>
</head>
<body>
<header>
  <h1>VIDEX <span>Native Cinema</span></h1>
  <div class="badge">100% software-rendered · no image assets</div>
  <div class="tag">SDF ray marching · {title}</div>
</header>
<div class="stage"><video src="/media/{file}" controls autoplay muted loop playsinline></video></div>
<div class="bar">
  <div class="tabs">{tabs}</div>
  <a class="dl" href="/media/{file}" download>Download MP4</a>
</div>
<div class="meta">
  <div class="cell"><div class="k">Film</div><div class="v">{title}</div></div>
  <div class="cell"><div class="k">Renderer</div><div class="v">videx_raytrace.py</div></div>
  <div class="cell"><div class="k">Frames</div><div class="v">{frames} @ 24 fps</div></div>
  <div class="cell"><div class="k">Size</div><div class="v">{size} MB</div></div>
</div>
</body>
</html>
"""


def films() -> list[Path]:
    if not OUTPUT.is_dir():
        return []
    found = sorted(OUTPUT.glob("*.mp4"))
    preferred = ["tokyo_seed_courier.mp4", "motion_graphics_test.mp4", "earth_from_space.mp4"]
    found.sort(key=lambda p: (preferred.index(p.name) if p.name in preferred else 99, p.name))
    return found


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        pass

    def _html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            items = films()
            if not items:
                self._html("<h1>No rendered films in output/</h1>")
                return
            current = items[0].name
            q = re.search(r"v=([\w.\-]+)", self.path)
            if q:
                asked = q.group(1)
                if any(p.name == asked for p in items):
                    current = asked
            tabs = "".join(
                f'<a class="tab{" active" if p.name == current else ""}" href="/?v={p.name}">'
                f'{html.escape(p.stem.replace("_", " ").title())}</a>'
                for p in items
            )
            chosen = next(p for p in items if p.name == current)
            size = chosen.stat().st_size / 1e6
            self._html(
                PAGE.format(
                    ar="2.39" if "tokyo" in current else "1.7778",
                    title=html.escape(chosen.stem.replace("_", " ").title()),
                    file=html.escape(current),
                    tabs=tabs,
                    frames="360" if "tokyo" in current else "—",
                    size=f"{size:.1f}",
                )
            )
            return

        if path.startswith("/media/"):
            name = Path(path[len("/media/"):]).name
            target = OUTPUT / name
            if not target.is_file():
                self.send_error(404)
                return
            self._stream(target)
            return

        self.send_error(404)

    def _stream(self, target: Path) -> None:
        size = target.stat().st_size
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                partial = True

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with target.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(262144, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"VIDEX Native Cinema Player on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
