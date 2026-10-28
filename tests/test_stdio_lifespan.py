import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from agent_helper.stdio import Lifespan, serve


def test_failed_fastapi_startup_does_not_read_requests():
    @asynccontextmanager
    async def lifespan(app):
        raise RuntimeError("startup failed")
        yield

    reads = []

    async def scenario():
        with pytest.raises(RuntimeError, match="startup failed"):
            await serve(FastAPI(lifespan=lifespan), "testserver", lambda: reads.append(True) or b"", lambda _: None)

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))
    assert reads == []


@pytest.mark.parametrize("failure", ["raise", "return", "failed-message", "failed-message-wait"])
def test_startup_termination_is_reported(failure):
    async def app(scope, receive, send):
        assert (await receive())["type"] == "lifespan.startup"
        if failure == "raise":
            raise RuntimeError("startup failed")
        if failure.startswith("failed-message"):
            await send({"type": "lifespan.startup.failed", "message": "startup failed"})
        if failure == "failed-message-wait":
            await asyncio.Event().wait()

    async def scenario():
        lifespan = Lifespan(app)
        with pytest.raises(RuntimeError, match="startup"):
            async with lifespan:
                pytest.fail("the body must not run after failed startup")
        assert lifespan.task.done()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))


@pytest.mark.parametrize("failure", ["raise", "failed-message", "failed-message-wait"])
def test_shutdown_failure_is_reported(failure):
    async def app(scope, receive, send):
        assert (await receive())["type"] == "lifespan.startup"
        await send({"type": "lifespan.startup.complete"})
        assert (await receive())["type"] == "lifespan.shutdown"
        if failure == "raise":
            raise RuntimeError("shutdown failed")
        await send({"type": "lifespan.shutdown.failed", "message": "shutdown failed"})
        if failure == "failed-message-wait":
            await asyncio.Event().wait()

    async def scenario():
        lifespan = Lifespan(app)
        with pytest.raises(RuntimeError, match="shutdown failed"):
            async with lifespan:
                pass
        assert lifespan.task.done()

    asyncio.run(asyncio.wait_for(scenario(), timeout=1))
