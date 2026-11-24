from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def cli(name, callback):
    @hydra.main(version_base="1.3", config_path=str(ROOT / "configs"), config_name=name)
    def main(cfg):
        return callback(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

    main()
