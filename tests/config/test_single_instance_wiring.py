"""One object per stateful node, not two.

``instantiate(cfg, _recursive_=True)`` builds a ``_target_`` dict wherever it
finds one. A plain ``${node}`` interpolation copies the *config* into the
consumer, so the node is built twice: once as the top-level definition, once
inside whoever referenced it. For a pure value object that is waste; for a
node that subscribes to the global event bus in ``__init__`` it is a bug.

Measured on the 2026-07-31 ablations before the fix: two live
``CostMonitorHook`` instances per run, each flushing its own
``[CostMonitorHookJSON]`` line for the same ``(mutant, attempts)``, each
running its own ACI update and its own agent budget — 28% of agent wakeups
were the twin handling an attempt the other had already handled, with two
``CostMonitorAgent`` LLM calls dispatched ~6 ms apart returning different
decisions.

``${ref:node}`` (gigaevo/config/resolvers.py) instantiates once and writes the
instance back into the config, so every later reference — and the top-level
walk itself — sees the same object.
"""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from gigaevo.config.resolvers import register_resolvers

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

# Engine-wiring keys that must share one instance with the top-level node:
# they own state, subscribe to the global bus, or are stopped by run.py as
# well as by the engine.
STATEFUL_KEYS = (
    "writer",
    "metrics_tracker",
    "pre_step_hook",
    "post_step_hook",
    "post_run_hook",
    "storage",
)


class _Counted:
    built: list[int] = []

    def __init__(self, x: int) -> None:
        _Counted.built.append(id(self))
        self.x = x


def _build(reference: str) -> tuple[int, bool]:
    """Instantiate a two-node config where `consumer` points at `node`.

    Returns (number of constructions, consumer got the same object).
    """
    register_resolvers()
    _Counted.built.clear()
    cfg = OmegaConf.create({
        "node": {"_target_": f"{__name__}._Counted", "x": 1},
        "consumer": {"_target_": "builtins.dict", "node": reference},
    })
    out = instantiate(cfg, _recursive_=True)
    shared = out["consumer"]["node"] is out["node"]
    return len(_Counted.built), shared


def test_plain_interpolation_builds_the_node_twice():
    """The defect itself — this is what `${post_step_hook}` was doing."""
    n, shared = _build("${node}")
    assert n == 2
    assert not shared


def test_ref_interpolation_builds_once_and_shares():
    n, shared = _build("${ref:node}")
    assert n == 1
    assert shared


def test_engine_wiring_uses_ref_for_stateful_nodes():
    """Guards the real configs against the plain-interpolation regression.

    Read raw (``resolve=False``): we are asserting on the interpolation
    syntax, not on what it resolves to, and resolving would need a Redis.
    """
    for name in ("default.yaml", "steady_state.yaml"):
        cfg = OmegaConf.load(CONFIG_DIR / "evolution" / name)
        for block in ("evolution_engine", "metrics_tracker"):
            node = cfg.get(block)
            if node is None:
                continue
            for key in STATEFUL_KEYS:
                raw = node._get_node(key)
                if raw is None:
                    continue
                raw = raw._value()
                if not (isinstance(raw, str) and raw.startswith("${")):
                    continue
                assert raw.startswith("${ref:"), (
                    f"evolution/{name}: {block}.{key} = {raw} — a plain "
                    f"interpolation builds a second instance; use ${{ref:...}}"
                )
