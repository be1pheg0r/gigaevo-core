from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[2] / "config"


def _compose(arm: str, *overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="config",
            overrides=[
                "problem.name=toy_kadane",
                f"+cost_monitor={arm}",
                *overrides,
            ],
        )


def test_cost_monitor_arms_share_estimator_and_controller_settings() -> None:
    enabled = _compose("enabled")
    agentless = _compose("agentless")

    assert enabled.cost_prediction == agentless.cost_prediction
    enabled_hook = OmegaConf.to_container(enabled.post_step_hook, resolve=False)
    agentless_hook = OmegaConf.to_container(agentless.post_step_hook, resolve=False)
    assert isinstance(enabled_hook, dict)
    assert isinstance(agentless_hook, dict)
    enabled_hook.pop("agent")
    agentless_hook.pop("agent")
    assert enabled_hook == agentless_hook
    assert enabled.cost_monitor_agent._target_.endswith("CostMonitorAgent")
    assert agentless.cost_monitor_agent._target_.endswith("NoOpCostMonitorAgent")


def test_cost_monitor_magic_constants_are_hydra_overridable() -> None:
    cfg = _compose(
        "agentless",
        "post_step_hook.aci_gamma=0.3",
        "post_step_hook.max_relative_step=0.2",
        "post_step_hook.concurrency_min_window_s=120",
        "cost_prediction.settings.warmup_calls=7",
        "max_in_flight=11",
        "evolution=steady_state",
    )

    assert cfg.post_step_hook.aci_gamma == 0.3
    assert cfg.post_step_hook.max_relative_step == 0.2
    assert cfg.post_step_hook.concurrency_min_window_s == 120
    assert cfg.cost_prediction.settings.warmup_calls == 7
    assert cfg.engine_config.max_in_flight == 11
