"""Dispatch a swarm of twin calls and guarantee they die with the request.

WHY THIS EXISTS. Every streaming study fans out with `asyncio.create_task` and
then walks `as_completed`, yielding a frame per twin. That shape has a hole in
it that only shows up when nobody is looking: if the consumer stops iterating —
the researcher navigated away, closed the tab, hit Run a second time, or the
proxy timed the response out — the `yield` raises `GeneratorExit` and the
generator unwinds. The tasks it created do not. They are still scheduled, still
holding the semaphore, and still calling OpenAI, for a result that now has
nowhere to go.

Measured shape of the problem before this: `swarm.py` and `boardroom.py` had a
`finally` that cancelled; `neuro.py`, `virality.py`, `optimizer.py`,
`sequence.py` and `studies.py` did not — five of seven study surfaces, including
the content study, the price curve and the objection radar. Combined with a
frontend that never aborted its `fetch`, the practical position was that a study
could not be stopped at all once started. It ran to completion and billed for
it whether or not a human was still there.

The fix is one helper rather than five copies of a `try/finally`, because the
next study added will copy whatever the last one did, and a cancellation
guarantee that depends on remembering is not a guarantee.

NOT A CANCELLATION OF WORK ALREADY PAID FOR. A twin whose HTTP request has
already been sent has already cost money; cancelling only stops the ones that
have not started and stops us waiting on the ones in flight. The saving is
proportional to how early the reader left, which is exactly when it is largest —
the common case is closing a tab in the first seconds of a forty-twin run.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Awaitable, Callable, Iterable, Optional, TypeVar

T = TypeVar("T")


async def as_they_land(
    coros: Iterable[Awaitable[T]],
    *,
    on_cancel: Optional[Callable[[int], None]] = None,
) -> AsyncIterator[T]:
    """Yield each result as it completes, cancelling the rest if the consumer stops.

    A drop-in for `for x in asyncio.as_completed(tasks)` that closes the hole
    described above. The caller keeps its own accounting — this decides nothing
    about what a result means, only that no task outlives the loop.

    `on_cancel` receives the number of tasks that were still outstanding, for
    logging. It is called only when cancellation actually happens, so a quiet
    log line means every twin was allowed to finish.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        for completed in asyncio.as_completed(tasks):
            yield await completed
    finally:
        # GeneratorExit, CancelledError, or an exception from the body — all
        # three land here, and all three mean the same thing: nobody is reading
        # any more.
        outstanding = [t for t in tasks if not t.done()]
        for task in outstanding:
            task.cancel()
        if outstanding:
            if on_cancel is not None:
                on_cancel(len(outstanding))
            # Wait for the cancellations to actually take effect before the
            # generator finishes unwinding. Without this the tasks are merely
            # *asked* to stop and the event loop may be torn down first, which
            # produces "Task was destroyed but it is pending!" and, worse,
            # leaves the semaphore permanently short of the slots they held.
            await asyncio.gather(*outstanding, return_exceptions=True)
