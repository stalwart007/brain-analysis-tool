"""Entrypoint that listens on IPv4 AND IPv6 from a single process.

Neither uvicorn CLI host value does both, and this deployment needs both:

  --host 0.0.0.0   IPv4 only. Fly's private network (6PN) is IPv6, so
                   `cogniswarm-api.internal:8000` resolves to fdaa:… and finds
                   nothing listening. The dashboard's ingest passthrough
                   returns 502 while the app looks perfectly healthy.

  --host ::        IPv6 only — uvicorn sets IPV6_V6ONLY even where the kernel
                   has bindv6only=0. Private networking then works, but Fly's
                   machine health check probes over IPv4 and gets connection
                   refused, so the machine is marked critical and deploys hang
                   waiting for a check that can never pass.

Both were observed on this app, in that order. A dual-stack socket — one
listener, V6ONLY cleared, IPv4 arriving as IPv4-mapped addresses — serves the
health check and the private network at once, and keeps working unchanged
under Docker Compose, whose default bridge is IPv4.
"""

from __future__ import annotations

import os
import socket
import sys

import uvicorn

PORT = int(os.environ.get("PORT", "8000"))


def _listener() -> socket.socket:
    if socket.has_dualstack_ipv6():
        return socket.create_server(
            ("::", PORT), family=socket.AF_INET6, dualstack_ipv6=True, backlog=128
        )
    # No dual-stack support (unusual on Linux, possible in restricted
    # sandboxes). IPv4 is the safer single choice: it keeps the health check
    # and Compose working, and the failure is then loud on the private network
    # rather than silent everywhere.
    print("[serve] dual-stack unavailable — binding IPv4 only", file=sys.stderr)
    return socket.create_server(("0.0.0.0", PORT), backlog=128)


if __name__ == "__main__":
    sock = _listener()
    # Single worker, deliberately: jobs.py re-queues every queued+running job
    # at startup with no compare-and-swap claim, so a second worker executes
    # every job a second time and bills OpenAI twice for it. SQLite has one
    # writer and the job worker is an in-process thread.
    uvicorn.run("app.main:app", fd=sock.fileno(), workers=1, access_log=True)
