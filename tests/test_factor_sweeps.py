import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mechanisms.config import ROOT
from mechanisms.sweep import expand


@pytest.mark.parametrize(
    "experiment, count",
    [
        ("discoverability", 110),
        ("detection", 60),
        ("payoff", 30),
        ("time_pressure", 60),
    ],
)
def test_presets(experiment, count):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = OmegaConf.to_container(
            compose(config_name="sweep", overrides=[f"experiment={experiment}"]),
            resolve=True,
        )
    trials = list(expand(config))
    assert len(trials) == count
    assert all(
        spec["treatment"]["exposure"]["kind"] == "nested_api" for spec, _, _ in trials
    )
    assert len({seed for _, seed, _ in trials}) == 10
