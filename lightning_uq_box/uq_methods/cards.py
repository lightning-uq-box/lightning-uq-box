# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""CARD Regression Diffusion Model.

Based on official PyTorch implementation from https://github.com/XzwHan/CARD # noqa: E501
"""

import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from .base import BaseModule
from .utils import (
    _get_num_outputs,
    default_classification_metrics,
    default_regression_metrics,
    process_classification_prediction,
    save_classification_predictions,
    save_regression_predictions,
)


# TODO check EMA support
# Support classification
class CARDBase(BaseModule):
    """CARD Model.

    Diffusion Model based on CARD paper.

    If you use this in your research, please cite the following paper:

    * https://arxiv.org/abs/2206.07275
    """

    pred_file_name = "predictions.csv"

    def __init__(
        self,
        cond_mean_model: nn.Module,
        guidance_model: nn.Module,
        n_steps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-5,
        beta_end: float = 1e-2,
        n_z_samples: int = 100,
        guidance_optim: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of the CARD Model.

        Args:
            cond_mean_model: conditional mean model, should be
                pretrained model that estimates $E[y|x]$
            guidance_model: guidance diffusion model
            n_steps: number of diffusion steps
            beta_schedule: what type of noise scheduling to conduct
            beta_start: start value of beta scheduling
            beta_end: end value of beta scheduling
            n_z_samples: number of samples during prediction
            guidance_optim: optimizer for the guidance model
            lr_scheduler: learning rate scheduler
        """
        super().__init__()

        self.cond_mean_model = cond_mean_model
        self.guidance_model = guidance_model
        self.n_steps = n_steps
        self.n_z_samples = n_z_samples

        self.noise_scheduler = NoiseScheduler(
            beta_schedule, n_steps, beta_start, beta_end
        )

        self.guidance_optim = guidance_optim
        self.lr_scheduler = lr_scheduler

        self.setup_task()

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        pass

    def diffusion_process(self, batch: dict[str, Tensor]) -> Tensor:
        """Diffusion process during training.

        Args:
            batch: the output of your DataLoader

        Returns:
            loss from diffusion process
        """
        x, y = batch[self.input_key], batch[self.target_key].float()

        batch_size = x.shape[0]

        # antithetic sampling
        ant_samples_t = torch.randint(
            low=0, high=self.n_steps, size=(batch_size // 2 + 1,)
        ).to(x.device)
        ant_samples_t = torch.cat(
            [ant_samples_t, self.n_steps - 1 - ant_samples_t], dim=0
        )[:batch_size]

        # noise estimation loss
        y_0_hat = self.cond_mean_model(x)

        e = torch.randn_like(y)
        y_t_sample = self.q_sample(
            y,
            y_0_hat,
            self.noise_scheduler.alphas_bar_sqrt.to(self.device),
            self.noise_scheduler.one_minus_alphas_bar_sqrt.to(self.device),
            ant_samples_t,
            noise=e,
        )

        guidance_output = self.guidance_model(x, y_t_sample, y_0_hat, ant_samples_t)

        # in classification y usually don't have target dimension
        # but in regression they do so for broadcasting align them
        if e.dim() == 1:
            e = e.unsqueeze(-1)
        # TODO does this change?
        # use the same noise sample e during training to compute loss
        loss = (e - guidance_output).square().mean()

        return loss, y_t_sample

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            training loss
        """
        loss, y_t_sample = self.diffusion_process(batch)

        self.log("train_loss", loss, batch_size=batch[self.input_key].shape[0])
        return loss

    # TODO what metrics should be logged?
    # def on_train_epoch_end(self):
    #     """Log epoch-level training metrics."""
    #     self.log_dict(self.train_metrics.compute())
    #     self.train_metrics.reset()

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the validation loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            validation loss
        """
        loss, y_t_sample = self.diffusion_process(batch)
        self.log("val_loss", loss, batch_size=batch[self.input_key].shape[0])
        return loss

    # def on_validation_epoch_end(self) -> None:
    #     """Log epoch level validation metrics."""
    #     self.log_dict(self.val_metrics.compute())
    #     self.val_metrics.reset()

    # def on_test_epoch_end(self):
    #     """Log epoch-level test metrics."""
    #     self.log_dict(self.test_metrics.compute())
    #     self.test_metrics.reset()

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            diffusion samples for each time step
        """
        # compute y_0_hat only once as the initial prediction
        with torch.no_grad():
            y_0_hat = self.cond_mean_model(X)

            if X.dim() == 2:
                # TODO: This works for Vector 1D Regression with the tiling
                # y_0_tile = torch.tile(y, (n_z_samples, 1))
                y_0_hat_tile = torch.tile(y_0_hat, (self.n_z_samples, 1)).to(
                    self.device
                )
                test_x_tile = torch.tile(X, (self.n_z_samples, 1)).to(self.device)

                z = torch.randn_like(y_0_hat_tile).to(self.device)

                # TODO check what happens, here and why y_0_hat_tile is passed twice
                y_t = y_0_hat_tile + z

                # generate samples from all time steps for the mini-batch
                y_tile_seq: list[Tensor] = self.p_sample_loop(
                    test_x_tile,
                    y_0_hat_tile,
                    y_0_hat_tile,
                    self.n_steps,
                    self.noise_scheduler.alphas.to(self.device),
                    self.noise_scheduler.one_minus_alphas_bar_sqrt.to(self.device),
                )

                # put in shape [n_z_samples, batch_size, output_dimension]
                y_tile_seq = [
                    arr.reshape(self.n_z_samples, X.shape[0], y_t.shape[-1])
                    for arr in y_tile_seq
                ]

                final_recoverd = y_tile_seq[-1]

            else:
                # TODO make this more efficient
                y_tile_seq: list[Tensor] = [
                    self.p_sample_loop(
                        X,
                        y_0_hat,
                        y_0_hat,
                        self.n_steps,
                        self.noise_scheduler.alphas.to(self.device),
                        self.noise_scheduler.one_minus_alphas_bar_sqrt.to(self.device),
                    )[-1]
                    for i in range(self.n_z_samples)
                ]

                final_recoverd = torch.stack(y_tile_seq, dim=0)

        return final_recoverd, y_tile_seq

    def p_sample(
        self,
        x: Tensor,
        y: Tensor,
        y_0_hat: Tensor,
        y_T_mean: Tensor,
        t: int,
        alphas: Tensor,
        one_minus_alphas_bar_sqrt: Tensor,
    ) -> Tensor:
        """Reverse diffusion process sampling, one time step.

        This is the process of generating a sample from the model's prior distribution
        and then evolving it through the diffusion process. It starts from the final
        time step and goes backwards to the initial time step. At each time step,
        a noise variable is sampled and the state is updated according to the
        reverse diffusion process.

        Args:
            x: input features
            y: sampled y at time step t, y_t.
            y_0_hat: prediction of pre-trained guidance model.
            y_T_mean: mean of prior distribution at timestep T.
            t: time step
            alphas: noise schedule alpha
            one_minus_alphas_bar_sqrt: noise schedule one minus alpha sqrt

        Returns:
            reverse process sample
        """
        z = torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
        t = torch.tensor([t]).to(self.device)
        alpha_t = self.extract(alphas, t, y)
        sqrt_one_minus_alpha_bar_t = self.extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_one_minus_alpha_bar_t_m_1 = self.extract(
            one_minus_alphas_bar_sqrt, t - 1, y
        )
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
        # y_t_m_1 posterior mean component coefficients
        gamma_0 = (
            (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        )
        gamma_1 = (
            (sqrt_one_minus_alpha_bar_t_m_1.square())
            * (alpha_t.sqrt())
            / (sqrt_one_minus_alpha_bar_t.square())
        )
        gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (
            alpha_t.sqrt() + sqrt_alpha_bar_t_m_1
        ) / (sqrt_one_minus_alpha_bar_t.square())
        eps_theta = self.guidance_model(x, y, y_0_hat, t).detach()
        # y_0 reparameterization
        y_0_reparam = (
            1
            / sqrt_alpha_bar_t
            * (
                y
                - (1 - sqrt_alpha_bar_t) * y_T_mean
                - eps_theta * sqrt_one_minus_alpha_bar_t
            )
        )
        # posterior mean
        y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean

        # posterior variance
        beta_t_hat = (
            (sqrt_one_minus_alpha_bar_t_m_1.square())
            / (sqrt_one_minus_alpha_bar_t.square())
            * (1 - alpha_t)
        )
        y_t_m_1 = y_t_m_1_hat.to(self.device) + beta_t_hat.sqrt().to(
            self.device
        ) * z.to(self.device)
        return y_t_m_1

    # Reverse function -- sample y_0 given y_1
    def p_sample_t_1to0(
        self,
        x: Tensor,
        y: Tensor,
        y_0_hat: Tensor,
        y_T_mean: Tensor,
        one_minus_alphas_bar_sqrt: Tensor,
    ) -> Tensor:
        """Reverse sample function, sample y_0 given y_1.

        Args:
            x: input
            y: sampled y at time step t, y_t.
            y_0_hat: prediction of pre-trained guidance model.
            y_T_mean: mean of prior distribution at timestep T.
            one_minus_alphas_bar_sqrt: noise schedule one minus alpha bar sqrt

        Returns:
            y_0 sample
        """
        # corresponding to timestep 1 (i.e., t=1 in diffusion models)
        t = torch.tensor([0]).to(self.device)
        sqrt_one_minus_alpha_bar_t = self.extract(one_minus_alphas_bar_sqrt, t, y)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        eps_theta = self.guidance_model(x, y, y_0_hat, t).detach()
        # y_0 reparameterization
        y_0_reparam = (
            1
            / sqrt_alpha_bar_t
            * (
                y
                - (1 - sqrt_alpha_bar_t) * y_T_mean
                - eps_theta * sqrt_one_minus_alpha_bar_t
            )
        )
        y_t_m_1 = y_0_reparam.to(self.device)
        return y_t_m_1

    def p_sample_loop(
        self,
        x: Tensor,
        y_0_hat: Tensor,
        y_T_mean: Tensor,
        n_steps: int,
        alphas: Tensor,
        one_minus_alphas_bar_sqrt: Tensor,
        only_last_sample: bool = False,
    ) -> list[Tensor]:
        """P sample loop for the entire chain.

        Args:
            x: input
            y_0_hat: prediction of pre-trained guidance model.
            y_T_mean: mean of prior distribution at timestep T.
            n_steps: number of diffusion steps
            alphas: noise schedule alpha
            one_minus_alphas_bar_sqrt: noise schedule one minus alpha
            only_last_sample: whether to only return the last sample

        Returns:
            list of samples for each diffusion time step
        """
        num_t, y_p_seq = None, None
        z = torch.randn_like(y_T_mean).to(self.device)
        cur_y = z + y_T_mean  # sampled y_T
        if only_last_sample:
            num_t = 1
        else:
            y_p_seq = [cur_y]
        for t in reversed(range(1, n_steps)):
            y_t = cur_y
            cur_y = self.p_sample(
                x, y_t, y_0_hat, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt
            )  # y_{t-1}
            if only_last_sample:
                num_t += 1
            else:
                y_p_seq.append(cur_y)
        if only_last_sample:
            assert num_t == n_steps
            y_0 = self.p_sample_t_1to0(
                x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt
            )
            return y_0
        else:
            assert len(y_p_seq) == n_steps
            y_0 = self.p_sample_t_1to0(
                x, y_p_seq[-1], y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt
            )
            y_p_seq.append(y_0)
            return y_p_seq

    def q_sample(
        self,
        y: Tensor,
        y_0_hat: Tensor,
        alphas_bar_sqrt: Tensor,
        one_minus_alphas_bar_sqrt: Tensor,
        t: int,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Q sampling process.

        This is the process of approximating the posterior distribution of the
        latent variables given the observed data. It starts from the initial
        time step and goes forward to the final time step. At each time step,
        a noise variable is sampled and the state is updated according to the
        forward diffusion process.

        Args:
            y: sampled y at time step t, y_t.
            y_0_hat: prediction of pre-trained guidance model.
            alphas_bar_sqrt: noise schedule alpha bar
            one_minus_alphas_bar_sqrt: noise schedule one minus alpha bar
            t: time step
            noise: optional noise tensor

        Returns:
            q sample at time step t
        """
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if noise is None:
            noise = torch.randn_like(y)
        elif noise.shape != y.shape:
            noise = noise.unsqueeze(-1)
        sqrt_alpha_bar_t = self.extract(alphas_bar_sqrt, t, y)
        sqrt_one_minus_alpha_bar_t = self.extract(one_minus_alphas_bar_sqrt, t, y)
        # q(y_t | y_0, x)
        # add feature dimension for proper broadcasting

        y_t = (
            sqrt_alpha_bar_t * y
            + (1 - sqrt_alpha_bar_t) * y_0_hat
            + sqrt_one_minus_alpha_bar_t * noise
        )
        return y_t

    def extract(self, input: Tensor, t: int, x: Tensor) -> Tensor:
        """Extract noise level at time step t from schedule.

        Args:
            input: noise input
            t: time step
            x: tensor to make noisy version of

        Returns:
            noisy version of x
        """
        shape = x.shape
        out = torch.gather(input, 0, t)
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute and return the test loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            test loss
        """
        out_dict = self.predict_step(batch[self.input_key])
        out_dict[self.target_key] = batch[self.target_key].detach().squeeze(-1).cpu()

        # turn mean to np array
        out_dict["pred"] = out_dict["pred"].detach().cpu().squeeze(-1)
        out_dict["pred_uct"] = out_dict["pred_uct"].detach().cpu().squeeze(-1)
        if "aleatoric_uct" in out_dict:
            out_dict["aleatoric_uct"] = (
                out_dict["aleatoric_uct"].detach().cpu().squeeze(-1)
            )

        # save metadata
        out_dict = self.add_aux_data_to_dict(out_dict, batch)

        return out_dict

    def configure_optimizers(self) -> Any:
        """Configure optimizers."""
        # lightning puts optimizer weights on device automatically
        optimizer = self.guidance_optim(self.guidance_model.parameters())

        # put conditional mean model on device as well
        self.cond_mean_model = self.cond_mean_model.to(self.device)

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class CARDRegression(CARDBase):
    """CARD Regression Model.

    If you use this in your research, please cite the following paper:

    * https://arxiv.org/abs/2206.07275
    """

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_regression_metrics("train")
        self.val_metrics = default_regression_metrics("val")
        self.test_metrics = default_regression_metrics("test")

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: batch index
            dataloader_idx: dataloader index

        Returns:
            prediction dictionary with uncertainty estimates and samples
        """
        final_recoverd, y_tile_seq = super().predict_step(X, batch_idx, dataloader_idx)

        # momenet matching
        mean_pred = final_recoverd.mean(dim=0).detach().cpu().squeeze()
        std_pred = final_recoverd.std(dim=0).detach().cpu().squeeze()

        return {
            "pred": mean_pred,
            "pred_uct": std_pred,
            "aleatoric_uct": std_pred,
            "samples": y_tile_seq,
        }

    def on_test_batch_end(
        self,
        outputs: dict[str, np.ndarray],
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """Test batch end save predictions."""
        del outputs["samples"]
        save_regression_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class CARDClassification(CARDBase):
    """CARD Classification Model.

    If you use this in your research, please cite the following paper:

    * https://arxiv.org/abs/2206.07275
    """

    valid_tasks = ["binary", "multiclass"]

    def __init__(
        self,
        cond_mean_model: nn.Module,
        guidance_model: nn.Module,
        n_steps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.00001,
        beta_end: float = 0.01,
        n_z_samples: int = 100,
        task: str = "multiclass",
        guidance_optim: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = None,
    ) -> None:
        """Initialize a new instance of the CARD Classification.

        Args:
            cond_mean_model: conditional mean model, should be
                pretrained model that estimates $E[y|x]$
            guidance_model: guidance diffusion model
            n_steps: number of diffusion steps
            beta_schedule: what type of noise scheduling to conduct
            beta_start: start value of beta scheduling
            beta_end: end value of beta scheduling
            n_z_samples: number of samples during prediction
            task: classification task, either `binary` or `multiclass`
            guidance_optim: optimizer for the guidance model
            lr_scheduler: learning rate scheduler
        """
        assert task in self.valid_tasks
        self.task = task

        self.num_classes = _get_num_outputs(cond_mean_model)

        super().__init__(
            cond_mean_model,
            guidance_model,
            n_steps,
            beta_schedule,
            beta_start,
            beta_end,
            n_z_samples,
            guidance_optim,
            lr_scheduler,
        )
        self.save_hyperparameters(
            ignore=[
                "cond_mean_model",
                "guidance_model",
                "guidance_optim",
                "lr_scheduler",
            ]
        )

    def setup_task(self) -> None:
        """Setup task specific attributes."""
        self.train_metrics = default_classification_metrics(
            "train", self.task, self.num_classes
        )
        self.val_metrics = default_classification_metrics(
            "val", self.task, self.num_classes
        )
        self.test_metrics = default_classification_metrics(
            "test", self.task, self.num_classes
        )

    def predict_step(
        self, X: Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Prediction step.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            predictions
        """
        final_recoverd, y_tile_seq = super().predict_step(X, batch_idx, dataloader_idx)
        # change from [num_samples, ...] to shape [batch_size, num_classes, num_samples]
        final_recoverd = final_recoverd.permute(1, 2, 0).cpu()

        # momenet matching
        pred_dict = process_classification_prediction(final_recoverd)
        pred_dict["samples"] = y_tile_seq

        return pred_dict

    def on_test_batch_end(
        self, outputs: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Test batch end save predictions.

        Args:
            outputs: dictionary of model outputs and aux variables
            batch_idx: batch index
            dataloader_idx: dataloader index
        """
        del outputs["samples"]
        save_classification_predictions(
            outputs, os.path.join(self.trainer.default_root_dir, self.pred_file_name)
        )


class NoiseScheduler:
    """Noise Scheduler for Diffusion Training."""

    valid_schedules = [
        "linear",
        "const",
        "quad",
        "jsd",
        "sigmoid",
        "cosine",
        "cosine_anneal",
    ]

    def __init__(
        self,
        schedule: str = "linear",
        n_steps: int = 1000,
        beta_start: float = 1e-5,
        beta_end: float = 1e-2,
    ) -> None:
        """Initialize a new instance of the noise scheduler.

        Args:
            schedule: type of noise schedule
            n_steps: number of diffusion time steps
            beta_start: beta noise start value
            beta_end: beta noise end value
        Raises:
            AssertionError if schedule is invalid
        """
        assert (
            schedule in self.valid_schedules
        ), f"Invalid schedule, please choose one of {self.valid_schedules}."
        self.schedule = schedule
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = {
            "linear": self.linear_schedule(),
            "const": self.constant_schedule(),
            "quad": self.quadratic_schedule(),
            "sigmoid": self.sigmoid_schedule(),
            "cosine": self.cosine_schedule(),
            "cosine_anneal": self.cosine_anneal_schedule(),
        }[schedule]

        self.betas_sqrt = torch.sqrt(self.betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)

    def linear_schedule(self) -> Tensor:
        """Linear Schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.n_steps)

    def constant_schedule(self) -> Tensor:
        """Constant Schedule."""
        return self.beta_end * torch.ones(self.n_steps)

    def quadratic_schedule(self) -> Tensor:
        """Quadratic Schedule."""
        return (
            torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.n_steps) ** 2
        )

    def sigmoid_schedule(self) -> Tensor:
        """Sigmoid Schedule."""
        betas = (
            torch.sigmoid(torch.linspace(-6, 6, self.n_steps))
            * (self.beta_end - self.beta_start)
            + self.beta_start
        )
        return torch.sigmoid(betas)

    def cosine_schedule(self) -> Tensor:
        """Cosine Schedule."""
        max_beta = 0.999
        cosine_s = 0.008
        return torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / self.n_steps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / self.n_steps + cosine_s) / (1 + cosine_s) * math.pi / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(self.n_steps)
            ]
        )

    def cosine_anneal_schedule(self) -> Tensor:
        """Cosine Annealing Schedule."""
        return torch.tensor(
            [
                self.beta_start
                + 0.5
                * (self.beta_end - self.beta_start)
                * (1 - math.cos(t / (self.n_steps - 1) * math.pi))
                for t in range(self.n_steps)
            ]
        )

    def get_noisy_x_at_t(input, t, x) -> Tensor:
        """Retrieve a noisy representation at time step t.

        Args:
            input: schedule version
            t: time step
            x: tensor ot make noisy version of

        Returns:
            A noisy
        """
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)
