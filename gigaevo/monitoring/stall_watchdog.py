"""Stop runs whose mutation counter remains stalled beyond a timeout."""

from __future__ import annotations

import asyncio
import threading
import time

from loguru import logger

from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.stopper import StopContext


def _check_stall(
    ctx: StopContext,
    last_mutants: int,
    stalled_since: float | None,
    now: float,
    stall_timeout_s: float,
) -> tuple[int, float | None, bool]:
    """Pure decision step. Returns (new_last_mutants, new_stalled_since,
    should_stop)."""
    if ctx.total_mutants != last_mutants:
        return ctx.total_mutants, None, False
    if ctx.elapsed_seconds <= 0:
        return last_mutants, stalled_since, False
    if stalled_since is None:
        return last_mutants, now, False
    return last_mutants, stalled_since, (now - stalled_since >= stall_timeout_s)


def _loop(
    engine: EvolutionEngine,
    loop: asyncio.AbstractEventLoop,
    stall_timeout_s: float,
    check_interval_s: float,
    stop: threading.Event,
) -> None:
    last_mutants = -1
    stalled_since: float | None = None
    while not stop.is_set():
        try:
            ctx = engine.build_stop_context()
            now = time.monotonic()
            last_mutants, stalled_since, should_stop = _check_stall(
                ctx, last_mutants, stalled_since, now, stall_timeout_s
            )
            if should_stop:
                logger.warning(
                    "[stall_watchdog] no mutant accepted for {:.0f}s "
                    "(total_mutants={}) — stopping run early instead of "
                    "spinning until the configured timeout",
                    now - stalled_since,
                    ctx.total_mutants,
                )
                asyncio.run_coroutine_threadsafe(engine.stop(), loop)
                break
        except Exception:
            logger.opt(exception=True).warning(
                "[stall_watchdog] check failed (will retry next interval)"
            )
        if stop.wait(check_interval_s):
            break


def start_stall_watchdog(
    engine: EvolutionEngine,
    *,
    stall_timeout_s: float = 300.0,
    check_interval_s: float = 15.0,
) -> threading.Event:
    """Start a daemon thread that stops ``engine`` if zero mutants are
    accepted for ``stall_timeout_s`` seconds. Returns the stop Event —
    call ``.set()`` to shut the watchdog down early (mirrors
    ``start_eta_ticker``).
    """
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    thread = threading.Thread(
        target=_loop,
        args=(engine, loop, stall_timeout_s, check_interval_s, stop),
        daemon=True,
        name="stall-watchdog",
    )
    thread.start()
    return stop
