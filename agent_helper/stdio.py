"""Serve the MCP endpoint over stdio: `python -m agent_helper.stdio` (docs/decisions/0031).

For clients and registries that start an MCP server as a local process and talk to it over stdin and
stdout (one JSON-RPC message per line) instead of over HTTP. Each line goes through the same app as a
POST to `/mcp` would, middleware included, so tools, limits and answers are those of the HTTP endpoint.
The process is its own instance with its own `DATA_DIR`; it does not reach any other instance. That is
enforced: push, relay and operator notifications are switched off, and it refuses to start on a database
that another instance (an HTTP server, or a second stdio process) uses.

Stdout carries only JSON-RPC messages; logs go to stderr.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import sqlite3
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit

from starlette.types import ASGIApp as App
from starlette.types import Message

from .app import create_app
from .config import Settings
from .mcp import META_VERSION

# The local process is the only client; it is counted and rate limited as one address.
PEER = ("127.0.0.1", 0)
OWNER_MARK = "served_by"  # instance_meta row: "stdio" once a stdio process created or used the database
LOCK_FILE = "stdio.lock"


class Refused(RuntimeError):
    """The process must not start (docs/decisions/0031)."""


class Oversized:
    """A line longer than the body limit; it is skipped, not read into memory."""


OVERSIZED = Oversized()


def stdio_settings(settings: Settings) -> Settings:
    """Nothing leaves this instance: no webhook to an operator, no push to agents, no relay, no jobs for a
    browser gateway. `stdio_mode` keeps the app from marking the database as an HTTP server's."""
    return dataclasses.replace(
        settings,
        notify_webhook_url=None,
        notify_webhook_secret=None,
        push_mode="off",
        relay_url=None,
        browse_mode="off",
        browse_gateway_url=None,
        browse_secret=None,
        stdio_mode=True,
    )


def line_reader(stream: IO[bytes], limit: int) -> Callable[[], bytes | Oversized]:
    """readline() with the same limit as an HTTP body. A longer line is drained in chunks and reported."""

    def read() -> bytes | Oversized:
        line = stream.readline(limit + 1)
        if len(line) <= limit or line.endswith(b"\n") and len(line) == limit + 1:
            return line
        while (rest := stream.readline(65536)) and not rest.endswith(b"\n"):
            pass
        return OVERSIZED

    return read


def claim_database(db_path: Path) -> Any:
    """Refuse a database that belongs to another instance, and hold a lock so a second stdio process on the
    same DATA_DIR is refused too. Returns the open lock file, to be kept open while serving."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(db_path.parent / LOCK_FILE, "a+b")  # noqa: SIM115 (held for the life of the process)
    try:
        _lock_exclusive(lock)
    except OSError:
        lock.close()
        raise Refused(f"another stdio process uses {db_path.parent}; give each one its own DATA_DIR") from None
    if db_path.exists() and db_path.stat().st_size > 0:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            try:
                row = db.execute("SELECT value FROM instance_meta WHERE name = ?", (OWNER_MARK,)).fetchone()
            except sqlite3.OperationalError:  # no such table: not a database of this service yet
                row = ("stdio",) if not db.execute("SELECT 1 FROM sqlite_master").fetchone() else None
        if row is None or row[0] != "stdio":
            lock.close()
            raise Refused(
                f"{db_path} belongs to another instance (an HTTP server); the stdio mode needs its own DATA_DIR"
            )
    return lock


def _lock_exclusive(lock: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def mark_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT OR IGNORE INTO instance_meta (name, value) VALUES (?, 'stdio')", (OWNER_MARK,))


def _header_value(value: str) -> str:
    """Non-ASCII header values travel in the Base64 sentinel form the endpoint decodes."""
    if value.isascii():
        return value
    return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode("ascii") + "?="


def headers_for(line: bytes, host: str) -> list[tuple[bytes, bytes]]:
    """The HTTP headers an HTTP client would send with this message. Modern MCP requires headers that
    repeat the protocol version, method and tool name from the body; over stdio there are none, so they
    are derived from the message itself."""
    headers = [
        (b"host", host.encode("ascii")),
        (b"content-type", b"application/json"),
        (b"accept", b"application/json, text/event-stream"),
    ]
    try:
        msg = json.loads(line)
    except (ValueError, UnicodeError, RecursionError):
        return headers  # the endpoint answers with its own parse error
    if not isinstance(msg, dict) or not isinstance(msg.get("method"), str):
        return headers
    params = msg.get("params")
    meta = params.get("_meta") if isinstance(params, dict) else None
    version = meta.get(META_VERSION) if isinstance(meta, dict) else None
    if isinstance(version, str) and version.isascii():
        headers.append((b"mcp-protocol-version", version.encode("ascii")))
        headers.append((b"mcp-method", _header_value(msg["method"]).encode("ascii")))
        name = params.get("name") if isinstance(params, dict) else None
        if msg["method"] == "tools/call" and isinstance(name, str):
            headers.append((b"mcp-name", _header_value(name).encode("ascii")))
    return headers


async def post(app: App, line: bytes, host: str) -> tuple[int, bytes]:
    """One in-process POST to /mcp; returns the status and the body."""
    scope: dict = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "root_path": "",
        "headers": [*headers_for(line, host), (b"content-length", str(len(line)).encode("ascii"))],
        "client": PEER,
        "server": (host, 80),
        "state": {},
    }
    sent = False
    status = 500
    body = bytearray()

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": line, "more_body": False}
        await asyncio.Event().wait()  # no disconnect while the call runs
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    await app(scope, receive, send)
    return status, bytes(body)


class Lifespan:
    """Runs the app's startup and shutdown, as a server would around serving requests."""

    def __init__(self, app: App) -> None:
        self.app = app
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.failure: str | None = None

    async def _receive(self) -> Message:
        return await self.inbox.get()

    async def _send(self, message: Message) -> None:
        if message["type"] in {"lifespan.startup.failed", "lifespan.shutdown.failed"}:
            self.failure = message.get("message") or message["type"]
        if message["type"].startswith("lifespan.startup"):
            self.started.set()
        elif message["type"].startswith("lifespan.shutdown"):
            self.stopped.set()

    async def __aenter__(self) -> Lifespan:
        scope = {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}

        async def run() -> None:
            try:
                await self.app(scope, self._receive, self._send)
            finally:
                # A task that exits without sending a result must also wake its waiter.
                self.started.set()
                self.stopped.set()

        self.task = asyncio.create_task(run())
        await self.inbox.put({"type": "lifespan.startup"})
        await self.started.wait()
        await self._raise_failure()
        if self.task.done():
            await self.task
            raise RuntimeError("application lifespan exited before startup completed")
        return self

    async def _raise_failure(self) -> None:
        if self.failure is not None:
            if self.task is not None:
                self.task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.task
            raise RuntimeError(self.failure)

    async def __aexit__(self, *_: object) -> None:
        await self.inbox.put({"type": "lifespan.shutdown"})
        await self.stopped.wait()
        await self._raise_failure()
        if self.task is not None:
            await self.task


async def serve(
    app: App, host: str, reader: Callable[[], bytes | Oversized], write: Callable[[bytes], None], limit: int = 0
) -> None:
    """Answer each line from `reader` until it returns b"" (end of input)."""
    loop = asyncio.get_running_loop()
    async with Lifespan(app):
        while line := await loop.run_in_executor(None, reader):
            if isinstance(line, Oversized):
                error = {"code": -32600, "message": f"message exceeds {limit} bytes"}
                write(json.dumps({"jsonrpc": "2.0", "id": None, "error": error}).encode() + b"\n")
                continue
            line = line.strip()
            if not line:
                continue
            _, body = await post(app, line, host)
            if body:  # notifications are answered with 202 and no body; stdio sends nothing back
                write(body.replace(b"\n", b"") + b"\n")


def main() -> None:
    settings = stdio_settings(Settings.from_env())
    host = urlsplit(settings.public_base_url).netloc or "localhost"
    try:
        lock = claim_database(settings.db_path)
    except Refused as e:
        print(f"agent-helper stdio: {e}", file=sys.stderr)
        sys.exit(2)

    def write(data: bytes) -> None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    with lock:
        app = create_app(settings)
        mark_database(settings.db_path)
        read = line_reader(sys.stdin.buffer, settings.max_body_bytes)
        asyncio.run(serve(app, host, read, write, settings.max_body_bytes))


if __name__ == "__main__":
    main()
