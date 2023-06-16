"""Run USA Vars OOD Script."""


import os
from typing import cast

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf
from setup_experiment import create_experiment_dir, generate_trainer


def set_up_omegaconf() -> DictConfig:
    """Loads program arguments from either YAML config files or command line arguments.

    This method loads defaults/a schema from "conf/defaults.yaml" as well as potential
    arguments from the command line. If one of the command line arguments is
    "config_file", then we additionally read arguments from that YAML file. One of the
    config file based arguments or command line arguments must specify task.name. The
    task.name value is used to grab a task specific defaults from its respective
    trainer. The final configuration is given as merge(task_defaults, defaults,
    config file, command line). The merge() works from the first argument to the last,
    replacing existing values with newer values. Additionally, if any values are
    merged into task_defaults without matching types, then there will be a runtime
    error.

    Returns:
        an OmegaConf DictConfig containing all the validated program arguments

    Raises:
        FileNotFoundError: when ``config_file`` does not exist
    """
    conf = OmegaConf.load("configs/usa_vars/gaussian_nll.yaml")
    command_line_conf = OmegaConf.from_cli()

    if "config_file" in command_line_conf:
        config_fn = command_line_conf.config_file
        if not os.path.isfile(config_fn):
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

        user_conf = OmegaConf.load(config_fn)
        conf = OmegaConf.merge(conf, user_conf)

    conf = OmegaConf.merge(  # Merge in any arguments passed via the command line
        conf, command_line_conf
    )
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright
    return conf


def main(conf: DictConfig) -> None:
    """Main training loop."""
    os.environ["HYDRA_FULL_ERROR"] = "1"
    torch.set_float32_matmul_precision("medium")

    exp_conf = create_experiment_dir(conf)

    with open(
        os.path.join(exp_conf["experiment"]["save_dir"], "config.yaml"), "w"
    ) as f:
        OmegaConf.save(config=conf, f=f)

    # Define module and datamodule
    datamodule: LightningDataModule = instantiate(conf.datamodule)
    model: LightningModule = instantiate(
        conf.uq_method, save_dir=conf.experiment.save_dir
    )

    trainer = generate_trainer(exp_conf)

    # run training
    trainer.fit(model=model, datamodule=datamodule)

    # test on IID
    trainer.test(ckpt_path="best", datamodule=datamodule)

    ood_splits = [
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (80, 90),
        (90, 100),
    ]

    # test on OOD
    for idx, ood_range in enumerate(ood_splits):
        # set pred file name
        model.pred_file_name = f"predictions_{ood_range[0]}_{ood_range[1]}.csv"

        trainer.test(ckpt_path="best", dataloaders=datamodule.ood_dataloader(ood_range))

    print("Finish Evaluation.")


if __name__ == "__main__":
    conf = set_up_omegaconf()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(conf.experiment.seed)

    # Main training procedure
    main(conf)
