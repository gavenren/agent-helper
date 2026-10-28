import asyncio
import io
import json
import os

import pytest

from agent_helper.app import create_app
from agent_helper.config import Settings
from agent_helper.stdio import (
    OVERSIZED,
    Refused,
    claim_database,
    line_reader,
    mark_database,
    serve,
    stdio_settings,
)

MODERN = "2026-07-28"
DIRECTORIES = [
    "normal",
    "research#1",
    pytest.param(
        "research?1",
        marks=pytest.mark.skipif(os.name == "nt", reason="question marks are not valid Windows filenames"),
    ),
    "research%23",
    "research notes",
    "研究",
]


def run(tmp_path, *messages):
    settings = Settings(public_base_url="http://testserver", data_dir=tmp_path)
    lines = [m if isinstance(m, bytes) else json.dumps(m).encode() + b"\n" for m in messages]
    lines.append(b"")
    out: list[bytes] = []
    asyncio.run(serve(create_app(settings), "testserver", lambda: lines.pop(0), out.append))
    assert all(chunk.endswith(b"\n") and chunk.count(b"\n") == 1 for chunk in out)
    return [json.loads(chunk) for chunk in out]


def test_legacy_handshake_and_tool_list(tmp_path):
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    }
    replies = run(
        tmp_path,
        init,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        b"\n",
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert [r["id"] for r in replies] == [1, 2]  # the notification and the blank line get no answer
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert replies[0]["result"]["serverInfo"]["name"] == "agent-helper"
    assert "find_help" in {t["name"] for t in replies[1]["result"]["tools"]}


def test_modern_call_gets_its_headers_from_the_message(tmp_path):
    meta = {
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "1"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    call = {
        "jsonrpc": "2.0",
        "id": "a",
        "method": "tools/call",
        "params": {"name": "find_help", "arguments": {"need": "a place to run a long job"}, "_meta": meta},
    }
    (reply,) = run(tmp_path, call)
    assert reply["id"] == "a"
    assert reply["result"]["isError"] is False


def test_unparseable_line_gets_a_parse_error(tmp_path):
    (reply,) = run(tmp_path, b"not json\n")
    assert reply["error"]["code"] == -32700


def test_nothing_leaves_a_stdio_instance(tmp_path):
    settings = Settings(
        public_base_url="http://testserver",
        data_dir=tmp_path,
        notify_webhook_url="https://hooks.example.invalid/x",
        notify_webhook_secret="s" * 32,
        push_mode="public",
        relay_url="https://relay.example.invalid",
        browse_mode="on",
        browse_gateway_url="http://browser.example.invalid:8080",
        browse_secret="g" * 32,
    )
    local = stdio_settings(settings)
    assert local.notify_webhook_url is None and local.push_mode == "off" and local.relay_url is None
    assert not local.push_enabled and not local.browse_enabled


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_a_database_of_another_instance_is_refused(tmp_path, directory):
    settings = Settings(public_base_url="http://testserver", data_dir=tmp_path / directory)
    create_app(settings)  # an HTTP server created this database
    with pytest.raises(Refused, match="another instance"):
        claim_database(settings.db_path)


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_a_new_or_own_database_is_accepted_but_only_once_at_a_time(tmp_path, directory):
    settings = Settings(public_base_url="http://testserver", data_dir=tmp_path / directory)
    with claim_database(settings.db_path):
        create_app(stdio_settings(settings))
        mark_database(settings.db_path)
        with pytest.raises(Refused, match="another stdio process"):
            claim_database(settings.db_path)
    with claim_database(settings.db_path):  # the same stdio instance, started again
        pass


def test_lines_are_limited_like_http_bodies(tmp_path):
    stream = io.BytesIO(b"x" * 50 + b"\n" + b"y" * 10 + b"\n" + b"z" * 9 + b"\n")
    read = line_reader(stream, 10)
    assert read() is OVERSIZED
    assert read() == b"y" * 10 + b"\n"
    assert read() == b"z" * 9 + b"\n"
    assert read() == b""

    settings = Settings(public_base_url="http://testserver", data_dir=tmp_path, max_body_bytes=64)
    stream = io.BytesIO(b'{"jsonrpc": "2.0", "id": 1, "method": "' + b"a" * 100 + b'"}\n')
    out: list[bytes] = []
    asyncio.run(serve(create_app(settings), "testserver", line_reader(stream, 64), out.append, 64))
    (reply,) = [json.loads(chunk) for chunk in out]
    assert reply["error"]["code"] == -32600 and reply["id"] is None


def test_a_server_started_on_a_stdio_database_takes_it_over(tmp_path):
    settings = Settings(public_base_url="http://testserver", data_dir=tmp_path)
    with claim_database(settings.db_path):
        create_app(stdio_settings(settings))
        mark_database(settings.db_path)
    create_app(settings)  # an HTTP server now uses this database
    with pytest.raises(Refused, match="another instance"):
        claim_database(settings.db_path)
