"""Tests for stall_watchdog module."""

from __future__ import annotations

import threading

from gigaevo.evolution.engine.stopper import StopContext
from gigaevo.monitoring.stall_watchdog import _check_stall


class TestCheckStall:
    def test_progress_resets_stall_clock(self):
        ctx = StopContext(total_mutants=5, elapsed_seconds=10.0)
        last_mutants, stalled_since, should_stop = _check_stall(
            ctx, last_mutants=4, stalled_since=100.0, now=200.0, stall_timeout_s=50.0
        )
        assert last_mutants == 5
        assert stalled_since is None
        assert should_stop is False

    def test_first_stall_tick_starts_the_clock(self):
        ctx = StopContext(total_mutants=0, elapsed_seconds=10.0)
        last_mutants, stalled_since, should_stop = _check_stall(
            ctx, last_mutants=0, stalled_since=None, now=200.0, stall_timeout_s=50.0
        )
        assert stalled_since == 200.0
        assert should_stop is False

    def test_stops_once_timeout_elapsed_with_no_progress(self):
        ctx = StopContext(total_mutants=0, elapsed_seconds=10.0)
        _, _, should_stop = _check_stall(
            ctx, last_mutants=0, stalled_since=100.0, now=151.0, stall_timeout_s=50.0
        )
        assert should_stop is True

    def test_does_not_stop_before_timeout_elapsed(self):
        ctx = StopContext(total_mutants=0, elapsed_seconds=10.0)
        _, _, should_stop = _check_stall(
            ctx, last_mutants=0, stalled_since=100.0, now=140.0, stall_timeout_s=50.0
        )
        assert should_stop is False

    def test_zero_elapsed_never_flagged_as_stalled(self):
        # Engine hasn't started ticking yet (elapsed_seconds<=0) — don't
        # start the stall clock on a cold start.
        ctx = StopContext(total_mutants=0, elapsed_seconds=0.0)
        _, stalled_since, should_stop = _check_stall(
            ctx, last_mutants=0, stalled_since=None, now=200.0, stall_timeout_s=50.0
        )
        assert stalled_since is None
        assert should_stop is False


class TestStartStallWatchdog:
    def test_returns_stop_event_and_is_stoppable(self):
        import asyncio
        from unittest.mock import MagicMock

        from gigaevo.monitoring.stall_watchdog import start_stall_watchdog

        async def _run():
            engine = MagicMock()
            engine.build_stop_context.return_value = StopContext(
                total_mutants=5, elapsed_seconds=10.0
            )
            stop = start_stall_watchdog(
                engine, stall_timeout_s=1000.0, check_interval_s=0.05
            )
            assert isinstance(stop, threading.Event)
            stop.set()

        asyncio.run(_run())
