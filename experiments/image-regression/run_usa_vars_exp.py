"""Run USA Vars OOD Script."""


import os

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf
from setup_experiment import create_experiment_dir, generate_trainer, set_up_omegaconf
from utils import ignore_args

from uq_method_box.datamodules.utils import collate_fn_laplace_torch


def train_from_scratch():
    pass


def train_from_scratch_and_fit_postprocess():
    pass


def fit_postprocess_from_ckpt():
    pass


def train_ensemble_from_scratch():
    pass


def fit_ensemble_from_ckpts():
    pass


def main(conf: DictConfig) -> None:
    """Main training loop."""
    torch.set_float32_matmul_precision("medium")

    exp_conf = create_experiment_dir(conf)

    with open(
        os.path.join(exp_conf["experiment"]["save_dir"], "config.yaml"), "w"
    ) as f:
        OmegaConf.save(config=conf, f=f)

    # Define module and datamodule
    datamodule: LightningDataModule = instantiate(conf.datamodule)
    trainer = generate_trainer(exp_conf)

    # run training
    if "post_processing" in conf:
        # import pdb
        # pdb.set_trace()
        base_model = instantiate(conf.uq_method, save_dir=conf.experiment.save_dir)
        state_dict = torch.load(conf.ckpt_path)["state_dict"]
        base_model.load_state_dict(state_dict)
        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        train_loader.collate_fn = collate_fn_laplace_torch

        model = instantiate(
            conf.post_processing,
            model=base_model.model,
            train_loader=train_loader,
            save_dir=conf.experiment.save_dir,
        )
        trainer.test(model=model, datamodule=datamodule)
    else:
        model: LightningModule = instantiate(
            conf.uq_method, save_dir=conf.experiment.save_dir
        )
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

        if "post_processing" in conf:
            trainer.test(model, dataloaders=datamodule.ood_dataloader(ood_range))
        else:
            trainer.test(
                ckpt_path="best", dataloaders=datamodule.ood_dataloader(ood_range)
            )

    print("Finish Evaluation.")


if __name__ == "__main__":
    conf = set_up_omegaconf()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(conf.experiment.seed)

    # Main training procedure
    main(conf)
