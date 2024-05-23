# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Test Image Diffusion Models Tasks."""

from pathlib import Path

import pytest
import torch
from denoising_diffusion_pytorch.repaint import GaussianDiffusion as RePaint
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch import Tensor

from lightning_uq_box.uq_methods.ddpm import DDPM, RePaintModel

model_config_paths = [
    "tests/configs/diffusion_models/ddpm.yaml",
    "tests/configs/diffusion_models/guided_ddpm.yaml",
    "tests/configs/diffusion_models/guidance_free_ddpm.yaml",
]

data_config_paths = ["tests/configs/image_classification/toy_classification.yaml"]

trainer_config_path = ["tests/configs/diffusion_models/trainer.yaml"]


class TestDDPMTasks:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    @pytest.mark.parametrize("trainer_config_path", trainer_config_path)
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        trainer_config_path: str,
        tmp_path: Path,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        trainer_conf = OmegaConf.load(trainer_config_path)

        full_conf = OmegaConf.merge(trainer_conf, data_conf, model_conf)

        model = instantiate(full_conf.model)

        datamodule = instantiate(full_conf.data)
        trainer = instantiate(
            full_conf.trainer,
            accelerator="cpu",
            default_root_dir=str(tmp_path),
            logger=CSVLogger(str(tmp_path)),
        )

        trainer.fit(model, datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)

        # test predict step with a single batch input
        batch = next(iter(datamodule.val_dataloader()))
        if "guidance_free" in model_config_path:
            sampled_images = model(batch["target"])
        else:
            sampled_images = model(batch["input"].shape[0])

        assert isinstance(sampled_images, Tensor)


class TestRePaint:
    @pytest.fixture
    def diffusion_model_and_data(
        self, tmp_path: Path
    ) -> tuple[DDPM, LightningDataModule]:
        model_config_path = "tests/configs/diffusion_models/ddpm.yaml"
        data_config_path = "tests/configs/image_classification/toy_classification.yaml"
        trainer_config_path = "tests/configs/diffusion_models/trainer.yaml"

        model_conf = OmegaConf.load(model_config_path)
        model = instantiate(model_conf.model)
        data_conf = OmegaConf.load(data_config_path)
        trainer_conf = OmegaConf.load(trainer_config_path)

        full_conf = OmegaConf.merge(trainer_conf, data_conf, model_conf)

        model = instantiate(full_conf.model)

        datamodule = instantiate(full_conf.data)
        trainer = instantiate(
            full_conf.trainer,
            accelerator="cpu",
            default_root_dir=str(tmp_path),
            logger=CSVLogger(str(tmp_path)),
        )

        trainer.fit(model, datamodule)
        return model, datamodule

    def test_repaint(
        self, diffusion_model_and_data: tuple[DDPM, LightningDataModule]
    ) -> None:
        diff_model, data = diffusion_model_and_data
        inpaint_model = RePaintModel(
            RePaint(
                diff_model.diffusion_model.model,
                image_size=diff_model.diffusion_model.image_size,
                timesteps=diff_model.diffusion_model.num_timesteps,
                sampling_timesteps=diff_model.diffusion_model.sampling_timesteps,
            )
        )

        batch = next(iter(data.train_dataloader()))

        def create_center_square_mask(image_size: int, mask_size: int):
            assert (
                image_size >= mask_size
            ), "Mask size should be smaller or equal to image size"

            mask = torch.zeros((image_size, image_size))
            start = (image_size - mask_size) // 2
            end = start + mask_size
            mask[start:end, start:end] = 1

            return (mask - 1) * -1

        image_size = batch["input"].shape[-1]

        mask_size = image_size // 3
        masks = (
            create_center_square_mask(image_size, mask_size)
            .repeat(batch["input"].shape[0], 1, 1)
            .unsqueeze(1)
        )

        # Apply the mask to the image
        masked_imgs = batch["input"] * masks

        pred_dict = inpaint_model.inpaint(masked_imgs, masks, num_samples=3)

        assert "pred_mean" in pred_dict
        assert "pred_uct" in pred_dict
        assert "samples" in pred_dict

        assert pred_dict["samples"].shape[1] == 3
