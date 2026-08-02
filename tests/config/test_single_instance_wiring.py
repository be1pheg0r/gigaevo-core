"""Verify that stateful config references resolve to one shared instance."""

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
    cfg = OmegaConf.create(
        {
            "node": {"_target_": f"{__name__}._Counted", "x": 1},
            "consumer": {"_target_": "builtins.dict", "node": reference},
        }
    )
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
