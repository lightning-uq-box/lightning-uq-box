"""Run USA Vars OOD Script."""


import os

import lightning.pytorch as pl
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf
from setup_experiment import create_experiment_dir, generate_trainer, set_up_omegaconf
from utils import ignore_args

from lightning_uq_box.datamodules.utils import collate_fn_laplace_torch


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

    def ood_collate(batch: dict[str, torch.Tensor]):
        """Collate fn to include augmentations."""
        try:
            images = [item["image"] for item in batch]
            labels = [item["labels"] for item in batch]
        except KeyError:
            images = [item["inputs"] for item in batch]
            labels = [item["targets"] for item in batch]

        lat = torch.stack([item["centroid_lat"] for item in batch])
        lon = torch.stack([item["centroid_lon"] for item in batch])

        # Stack images and labels into tensors
        inputs = torch.stack(images)
        targets = torch.stack(labels)

        return datamodule.on_after_batch_transfer(
            {
                "image": inputs,
                "labels": targets,
                "centroid_lat": lat,
                "centroid_lon": lon,
            },
            dataloader_idx=0,
        )

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
            save_dir=conf.experiment.save_dir,
        )
        trainer.test(model=model, datamodule=datamodule)

    elif "BNN_VI_ELBO" in conf["uq_method"]["_target_"]:
        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        model: LightningModule = instantiate(
            conf.uq_method,
            save_dir=conf["experiment"]["save_dir"],
            num_training_points=len(train_loader.dataset),
        )
        trainer.fit(model=model, datamodule=datamodule)
        # test on IID
        trainer.test(ckpt_path="best", datamodule=datamodule)

    else:
        model: LightningModule = instantiate(
            conf.uq_method, save_dir=conf.experiment.save_dir
        )
        trainer.fit(model=model, datamodule=datamodule)
        # test test
        trainer.test(ckpt_path="best", datamodule=datamodule)

    # save results for train and val predictions
    model.pred_file_name = f"predictions_train.csv"
    train_loader = datamodule.train_dataloader()
    train_loader.num_workers
    train_loader.collate_fn = ood_collate

    if "post_processing" in conf:
        trainer.test(model, dataloaders=train_loader)
    else:
        trainer.test(ckpt_path="best", dataloaders=train_loader)

    model.pred_file_name = f"predictions_val.csv"
    val_loader = datamodule.train_dataloader()
    val_loader.num_workers
    val_loader.collate_fn = ood_collate

    if "post_processing" in conf:
        trainer.test(model, dataloaders=train_loader)
    else:
        trainer.test(ckpt_path="best", dataloaders=train_loader)


if __name__ == "__main__":
    conf = set_up_omegaconf()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(conf.experiment.seed)

    # Main training procedure
    main(conf)
