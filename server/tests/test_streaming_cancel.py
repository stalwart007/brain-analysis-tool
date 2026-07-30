"""Cancellation: a study nobody is watching must stop costing money.

The hole this closes was real and expensive. Every streaming study fans out with
`create_task` and walks `as_completed`; if the consumer stops iterating — the
reader navigated away, closed the tab, or pressed Run again — the generator
unwinds and the tasks do not. They stay scheduled, keep the semaphore, and keep
calling OpenAI for a result with nowhere to go.

Five of seven study surfaces had no `finally` at all: the content study, the
price curve, the objection radar, virality, optimize and sequence. Combined with
a frontend that never aborted its fetch, a study could not be stopped once
started.
"""

import asyncio

import pytest

from app.streaming import as_they_land


def test_results_arrive_as_they_complete_not_in_submission_order():
    """The whole point of the fan-out: a fast twin is shown immediately rather
    than behind a slow one."""

    async def work(name, delay):
        await asyncio.sleep(delay)
        return name

    async def main():
        out = []
        async for r in as_they_land([work("slow", 0.05), work("fast", 0.001)]):
            out.append(r)
        return out

    assert asyncio.run(main()) == ["fast", "slow"]


def test_abandoning_the_loop_cancels_everything_still_running():
    """THE regression, and the one that costs money.

    Breaking out of the loop is what a client disconnect looks like from inside
    the generator. Every task that had not finished must be cancelled, and must
    have ACTUALLY stopped by the time the generator finishes unwinding — not
    merely been asked to.
    """
    started, finished = [], []

    async def work(i, delay):
        started.append(i)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        finished.append(i)
        return i

    cancelled_count = []

    async def main():
        # One returns immediately; four are still in flight when we walk away.
        agen = as_they_land(
            [work(0, 0.001)] + [work(i, 5.0) for i in range(1, 5)],
            on_cancel=cancelled_count.append,
        )
        async for _ in agen:
            break                      # the reader leaves after the first result
        await agen.aclose()            # what the runtime does on GeneratorExit

    asyncio.run(asyncio.wait_for(main(), timeout=2.0))

    assert cancelled_count == [4], "all four outstanding tasks should be cancelled"
    # None of the slow ones reached their body's end — they were stopped, not
    # merely abandoned to finish in the background.
    assert finished == [0]


def test_an_exception_in_the_consumer_also_cancels():
    """A panel that throws mid-stream must not leave twins running either."""
    cancelled = []

    async def work(delay):
        await asyncio.sleep(delay)
        return delay

    async def main():
        agen = as_they_land([work(0.001)] + [work(5.0) for _ in range(3)],
                            on_cancel=cancelled.append)
        with pytest.raises(RuntimeError):
            async for _ in agen:
                raise RuntimeError("panel blew up")
        await agen.aclose()

    asyncio.run(asyncio.wait_for(main(), timeout=2.0))
    assert cancelled and cancelled[0] == 3


def test_a_completed_run_cancels_nothing():
    """A quiet `on_cancel` is the signal that every twin was allowed to finish;
    a false alarm here would make the log useless."""
    cancelled = []

    async def work(i):
        await asyncio.sleep(0.001)
        return i

    async def main():
        out = []
        async for r in as_they_land([work(i) for i in range(4)], on_cancel=cancelled.append):
            out.append(r)
        return out

    assert sorted(asyncio.run(main())) == [0, 1, 2, 3]
    assert cancelled == []


def test_exceptions_from_a_task_still_reach_the_consumer():
    """Cancellation must not swallow real failures — a twin that raises is a
    twin the caller has to count as failed."""

    async def boom():
        raise ValueError("twin died")

    async def main():
        async for _ in as_they_land([boom()]):
            pass

    with pytest.raises(ValueError):
        asyncio.run(main())


@pytest.mark.parametrize(
    "module,fn",
    [
        ("app.neuro", "stream_content_study"),
        ("app.virality", None),
        ("app.optimizer", None),
        ("app.sequence", None),
        ("app.studies", None),
    ],
)
def test_every_study_module_dispatches_through_the_cancelling_helper(module, fn):
    """No study may go back to a bare `as_completed`.

    Written as a source check rather than a behavioural one because the failure
    is invisible at runtime — the study works perfectly, and only the bill
    shows it. The next study will be written by copying one of these.
    """
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module))
    assert "as_they_land" in src, f"{module} must dispatch through as_they_land"
    assert "asyncio.as_completed" not in src, (
        f"{module} still uses a bare as_completed, which orphans tasks when the "
        "client disconnects"
    )
