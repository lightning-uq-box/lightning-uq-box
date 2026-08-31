# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.
"""Test pixelwise regression task."""

import os
import re
from pathlib import Path
from typing import Any

import h5py
import pytest
import torch
from conftest import minimal_trainer_kwargs
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from pytest import TempPathFactory

from lightning_uq_box.datamodules import ToyPixelwiseRegressionDataModule
from lightning_uq_box.models.pixel_cnn import PixelCNN
from lightning_uq_box.uq_methods import DeepEnsemblePxRegression
from lightning_uq_box.uq_methods.loss_functions import VQVAELoss

seed_everything(0)

model_config_paths = [
    "tests/configs/pixelwise_regression/base.yaml",
    "tests/configs/pixelwise_regression/mve.yaml",
    "tests/configs/pixelwise_regression/der.yaml",
    "tests/configs/pixelwise_regression/quantile_regression.yaml",
    "tests/configs/pixelwise_regression/swag.yaml",
    "tests/configs/pixelwise_regression/vae_conv_encoder.yaml",
    "tests/configs/pixelwise_regression/vae_vit_encoder.yaml",
    "tests/configs/pixelwise_regression/vae_conditional.yaml",
    "tests/configs/pixelwise_regression/vqvae_conv_encoder.yaml",
    "tests/configs/pixelwise_regression/vqvae_vit_encoder.yaml",
]

data_config_paths = ["tests/configs/pixelwise_regression/toy_pixelwise_regression.yaml"]


class TestPixelwiseRegressionTask:
    @pytest.mark.parametrize("model_config_path", model_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        tmp_path: Path,
        accelerator_config: dict,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        full_conf = OmegaConf.merge(data_conf, model_conf)

        model = instantiate(full_conf.uq_method, save_preds=True)
        datamodule = instantiate(full_conf.data)
        trainer = Trainer(
            **minimal_trainer_kwargs(
                accelerator_config,
                tmp_path,
                max_epochs=2 if "swag" in model_config_path else 1,
            )
        )

        if "conformal" in model_config_path:
            trainer.validate(model, datamodule.calib_dataloader())
        else:
            trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

        with h5py.File(os.path.join(model.pred_dir, "batch_0_sample_0.hdf5"), "r") as f:
            assert "pred" in f
            assert "target" in f
            for key in ["pred", "target"]:
                assert f[key].shape[-1] == datamodule.image_size
                assert f[key].shape[-2] == datamodule.image_size
            assert "aux" in f.attrs
            assert "index" in f.attrs

    @pytest.mark.parametrize(
        "model_config_path",
        [
            "tests/configs/pixelwise_regression/base.yaml",
            "tests/configs/pixelwise_regression/mve.yaml",
            "tests/configs/pixelwise_regression/der.yaml",
            "tests/configs/pixelwise_regression/quantile_regression.yaml",
        ],
    )
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_predict_step(self, model_config_path: str, data_config_path: str) -> None:
        """Test predict step output shapes."""
        model_path_basename = os.path.basename(model_config_path)

        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        full_conf = OmegaConf.merge(data_conf, model_conf)

        batch_size = data_conf.data.get("batch_size", 4)

        expected_shapes = {
            "base.yaml": (batch_size, 1, 64, 64),
            "mve.yaml": (batch_size, 1, 64, 64),
            "der.yaml": (batch_size, 1, 64, 64),
            "quantile_regression.yaml": (batch_size, 64, 64),
        }

        model = instantiate(full_conf.uq_method)
        datamodule = instantiate(full_conf.data)

        val_loader = datamodule.val_dataloader()
        val_batch = next(iter(val_loader))

        pred_dict = model.predict_step(val_batch["input"])
        assert "pred" in pred_dict

        if model_path_basename in ["mve.yaml", "der.yaml"]:
            assert "pred_uct" in pred_dict
            assert pred_dict["pred"].shape == (batch_size, 1, 64, 64), (
                f"Failed for {model_config_path}"
            )
            assert (
                pred_dict["pred_uct"].shape
                == expected_shapes[os.path.basename(model_config_path)]
            ), f"Failed for {model_config_path}"
        elif model_path_basename in ["quantile_regression.yaml"]:
            assert "lower" in pred_dict
            assert "upper" in pred_dict
            assert pred_dict["pred"].shape == (batch_size, 64, 64), (
                f"Failed for {model_config_path}"
            )
            assert (
                pred_dict["lower"].shape
                == expected_shapes[os.path.basename(model_config_path)]
            ), f"Failed for {model_config_path}"
            assert (
                pred_dict["upper"].shape
                == expected_shapes[os.path.basename(model_config_path)]
            ), f"Failed for {model_config_path}"


mc_dropout_config_paths = ["tests/configs/pixelwise_regression/mc_dropout.yaml"]


class TestMCDropout:
    @pytest.mark.parametrize("model_config_path", mc_dropout_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        tmp_path: Path,
        accelerator_config: dict,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.uq_method)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(
            **minimal_trainer_kwargs(accelerator_config, tmp_path, checkpoints=True)
        )
        with pytest.raises(UserWarning, match="No dropout layers found in model"):
            trainer.fit(model, datamodule)
            trainer.test(ckpt_path="best", datamodule=datamodule)


ensemble_model_config_paths = ["tests/configs/pixelwise_regression/mve.yaml"]


class TestDeepEnsemble:
    @pytest.fixture(
        params=[
            (model_config_path, data_config_path)
            for model_config_path in ensemble_model_config_paths
            for data_config_path in data_config_paths
        ]
    )
    def ensemble_members_dict(
        self, request, tmp_path_factory: TempPathFactory, accelerator_config: dict
    ) -> list[dict[str, Any]]:
        model_config_path, data_config_path = request.param
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)
        # train networks for deep ensembles
        ckpt_paths = []
        for i in range(2):
            tmp_path = tmp_path_factory.mktemp(f"run_{i}")

            model = instantiate(model_conf.uq_method)
            datamodule = instantiate(data_conf.data)
            trainer = Trainer(
                **minimal_trainer_kwargs(accelerator_config, tmp_path, checkpoints=True)
            )
            trainer.fit(model, datamodule)
            ckpt_cb = trainer.checkpoint_callback
            assert isinstance(ckpt_cb, ModelCheckpoint)
            ckpt_file = ckpt_cb.best_model_path
            assert ckpt_file
            ckpt_paths.append({"base_model": model, "ckpt_path": ckpt_file})

        return ckpt_paths

    def test_deep_ensemble(
        self,
        ensemble_members_dict: list[dict[str, Any]],
        tmp_path: Path,
        accelerator_config: dict,
    ) -> None:
        """Test Deep Ensemble."""
        ensemble_model = DeepEnsemblePxRegression(
            ensemble_members_dict, save_preds=True
        )
        datamodule = ToyPixelwiseRegressionDataModule(num_images=2, batch_size=2)
        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))
        trainer.test(ensemble_model, datamodule=datamodule)

        # check that predictions are saved
        assert os.path.exists(ensemble_model.pred_dir)


posthoc_config_paths = [
    "tests/configs/pixelwise_regression/img2img_conformal.yaml",
    "tests/configs/pixelwise_regression/img2img_conformal_smp.yaml",
]


class TestPosthoc:
    @pytest.mark.parametrize("model_config_path", posthoc_config_paths)
    @pytest.mark.parametrize("data_config_path", data_config_paths)
    @pytest.mark.parametrize("calibration", [True, False])
    def test_trainer(
        self,
        model_config_path: str,
        data_config_path: str,
        calibration: bool,
        tmp_path: Path,
        accelerator_config: dict,
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        data_conf = OmegaConf.load(data_config_path)

        model = instantiate(model_conf.uq_method)
        datamodule = instantiate(data_conf.data)
        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))

        if calibration:
            trainer.fit(model, train_dataloaders=datamodule.calib_dataloader())
            trainer.test(model, datamodule=datamodule)
        else:
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Model has not been post hoc fitted, please call trainer.fit(model, train_dataloaders=dm.calib_dataloader()) first."
                ),
            ):
                X = torch.rand(1, 3, 64, 64)
                model.predict_step(X)


frozen_config_paths = [
    "tests/configs/pixelwise_regression/base.yaml",
    "tests/configs/pixelwise_regression/mc_dropout.yaml",
    "tests/configs/pixelwise_regression/quantile_regression.yaml",
    "tests/configs/pixelwise_regression/mve.yaml",
    "tests/configs/pixelwise_regression/der.yaml",
]


class TestFrozenPxRegression:
    @pytest.mark.parametrize("model_name", ["Unet", "DeepLabV3Plus"])
    @pytest.mark.parametrize(
        "backbone", ["resnet18", "tu-swin_tiny_patch4_window7_224"]
    )
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_backbone(
        self, model_config_path: str, model_name: str, backbone: str
    ) -> None:
        model_conf = OmegaConf.load(model_config_path)
        model_conf.uq_method.model["_target_"] = (
            f"segmentation_models_pytorch.{model_name}"
        )
        model_conf.uq_method.model["encoder_name"] = backbone

        if model_name == "DeepLabV3Plus":
            # drop depth and decoder_channels
            model_conf.uq_method.model.pop("encoder_depth")
            model_conf.uq_method.model.pop("decoder_channels")

        module = instantiate(model_conf.uq_method, freeze_backbone=True)
        seg_model = module.model

        assert all(
            param.requires_grad is False for param in seg_model.encoder.parameters()
        )
        assert all(param.requires_grad for param in seg_model.decoder.parameters())
        assert all(
            param.requires_grad for param in seg_model.segmentation_head.parameters()
        )

    @pytest.mark.parametrize("model_name", ["Unet", "DeepLabV3Plus"])
    @pytest.mark.parametrize("model_config_path", frozen_config_paths)
    def test_freeze_decoder(self, model_config_path: str, model_name: str) -> None:
        model_conf = OmegaConf.load(model_config_path)
        model_conf.uq_method.model["_target_"] = (
            f"segmentation_models_pytorch.{model_name}"
        )

        if model_name == "DeepLabV3Plus":
            # drop depth and decoder_channels
            model_conf.uq_method.model.pop("encoder_depth")
            model_conf.uq_method.model.pop("decoder_channels")

        module = instantiate(model_conf.uq_method, freeze_decoder=True)
        seg_model = module.model

        assert all(
            param.requires_grad is False for param in seg_model.decoder.parameters()
        )
        assert all(param.requires_grad for param in seg_model.encoder.parameters())
        assert all(
            param.requires_grad for param in seg_model.segmentation_head.parameters()
        )


frozen_vae_paths = [
    "tests/configs/pixelwise_regression/vae_conv_encoder.yaml",
    "tests/configs/pixelwise_regression/vae_vit_encoder.yaml",
    "tests/configs/pixelwise_regression/vae_conditional.yaml",
]

frozen_vqvae_paths = [
    "tests/configs/pixelwise_regression/vqvae_conv_encoder.yaml",
    "tests/configs/pixelwise_regression/vqvae_vit_encoder.yaml",
]


class TestFrozenVAE:
    @pytest.mark.parametrize("model_config_path", frozen_vae_paths)
    def test_freeze_encoder(self, model_config_path: str) -> None:
        model_conf = OmegaConf.load(model_config_path)
        module = instantiate(model_conf.uq_method, freeze_backbone=True)
        assert all(
            param.requires_grad is False for param in module.encoder.parameters()
        )

    @pytest.mark.parametrize("model_config_path", frozen_vae_paths)
    def test_freeze_decoder(self, model_config_path: str) -> None:
        model_conf = OmegaConf.load(model_config_path)
        module = instantiate(model_conf.uq_method, freeze_decoder=True)
        assert all(
            param.requires_grad is False for param in module.decoder.parameters()
        )


class TestFrozenVQVAE:
    @pytest.mark.parametrize("model_config_path", frozen_vqvae_paths)
    def test_freeze_encoder(self, model_config_path: str) -> None:
        model_conf = OmegaConf.load(model_config_path)
        module = instantiate(model_conf.uq_method, freeze_backbone=True)
        assert all(
            param.requires_grad is False for param in module.encoder.parameters()
        )

    @pytest.mark.parametrize("model_config_path", frozen_vqvae_paths)
    def test_freeze_decoder(self, model_config_path: str) -> None:
        model_conf = OmegaConf.load(model_config_path)
        module = instantiate(model_conf.uq_method, freeze_decoder=True)
        assert all(
            param.requires_grad is False for param in module.decoder.parameters()
        )


class TestVQVAE:
    """Tests for behaviour specific to the discrete latent."""

    @pytest.fixture
    def module(self) -> Any:
        model_conf = OmegaConf.load(
            "tests/configs/pixelwise_regression/vqvae_conv_encoder.yaml"
        )
        return instantiate(model_conf.uq_method)

    def test_latent_is_a_spatial_grid(self, module: Any) -> None:
        """The indices must form a grid, which is what makes a PixelCNN prior possible."""
        X = torch.randn(2, 3, module.img_size, module.img_size)
        quantized, indices, commit_loss = module.encode_img_to_latent(X)

        grid = module.latent_feature_dim
        assert indices.shape == (2, grid, grid)
        assert quantized.shape == (2, module.latent_channels, grid, grid)
        assert commit_loss.ndim == 0

    def test_indices_round_trip_to_quantized(self, module: Any) -> None:
        """Ancestral sampling depends on mapping indices back to codebook vectors."""
        module.eval()
        X = torch.randn(2, 3, module.img_size, module.img_size)
        with torch.no_grad():
            quantized, indices, _ = module.encode_img_to_latent(X)
            recovered = module.vq_module.get_output_from_indices(indices)

        assert torch.allclose(recovered, quantized, atol=1e-5)

    def test_commit_loss_is_zero_in_eval_mode(self, module: Any) -> None:
        """Pin a library behaviour that makes ``val_commit_loss`` uninformative.

        ``VectorQuantize`` computes the commitment loss only in training mode and
        returns exactly 0.0 in eval. That is why ``val_commit_loss`` is always
        0.0, ``val_loss`` equals ``val_rec_loss`` during validation, and the
        metric is excluded from seed aggregation. If a library upgrade ever
        starts computing it in eval, this test fails and the docs and the
        aggregation script need revisiting.
        """
        X = torch.randn(2, 3, module.img_size, module.img_size)

        module.train()
        _, _, commit_train = module.encode_img_to_latent(X)

        module.eval()
        with torch.no_grad():
            _, _, commit_eval = module.encode_img_to_latent(X)

        assert commit_train > 0.0, "commitment loss should be live in train mode"
        assert commit_eval == 0.0, "commitment loss is not computed in eval mode"

    def test_training_uses_deterministic_code_assignment(self, module: Any) -> None:
        """Training must quantize to the true nearest codebook entry.

        ``stochastic_sample_codes=True`` is gated on ``module.training``, so
        enabling it on the codebook perturbs the *training* assignment with
        Gumbel noise. The distance margin between the best and second-best code
        is around 0.01 while that noise has standard deviation ~1.28, so the
        assignment becomes very nearly uniform over the codebook and both the
        EMA update and the logged perplexity describe the noise rather than the
        model. Stochastic sampling belongs only in ``predict_step``, which
        rebinds it for the duration of the call.

        The encoder is bypassed here because its batch norm legitimately makes
        train and eval latents differ; this pins the quantizer alone.
        """
        latent = torch.randn(4, module.latent_channels, 2, 2)

        module.train()
        with torch.no_grad():
            _, indices_train, _ = module.vq_module(latent, freeze_codebook=True)

        module.eval()
        with torch.no_grad():
            _, indices_eval, _ = module.vq_module(latent)

        assert torch.equal(indices_train, indices_eval), (
            "training assignments differ from the deterministic nearest-neighbour "
            "assignment, which means codes are being sampled stochastically during "
            "training"
        )

    def test_quantizer_forward_is_exactly_a_codebook_entry(self, module: Any) -> None:
        """The quantized output must be the selected codebook vector, bit for bit.

        This is the defining property of vector quantization: anything that leaks
        the continuous encoder output into the forward pass would make the decoder
        train against a representation the ancestral sampler can never produce,
        since sampling reaches the decoder only through codebook indices. Stage 2
        would then degrade for reasons invisible in stage-1 metrics.
        """
        latent = torch.randn(4, module.latent_channels, 2, 2)
        module.eval()
        with torch.no_grad():
            quantized, indices, _ = module.vq_module(latent)

        expected = module.vq_module._codebook.embed[0][indices].permute(0, 3, 1, 2)
        assert torch.allclose(quantized, expected, atol=1e-5)

        # the same lookup the ancestral sampler uses must agree with the forward pass
        round_trip = module.vq_module.get_output_from_indices(indices)
        assert torch.allclose(round_trip, quantized, atol=1e-5), (
            "get_output_from_indices disagrees with the forward pass, so decoding "
            "sampled indices in stage 2 would not match training-time quantization"
        )

    def test_gradient_estimator_is_the_configured_one(self, module: Any) -> None:
        """The quantizer must use the gradient estimator that was asked for.

        vector_quantize_pytorch silently enables the rotation trick of Fifty et al.
        2024 for any ``dim > 1``, replacing the identity straight-through estimator
        of van den Oord et al. The forward pass is identical under both, so no
        reconstruction, shape, or codebook-usage test can detect the substitution --
        only the backward pass differs. This pins it to the declared setting.
        """
        assert module.vq_module.rotation_trick == module.rotation_trick

        latent = torch.randn(4, module.latent_channels, 2, 2, requires_grad=True)
        module.train()
        quantized, _, _ = module.vq_module(latent, freeze_codebook=True)
        grad = torch.autograd.grad(quantized.sum(), latent)[0]

        if module.rotation_trick:
            # the rotation trick reshapes the gradient, so identity would mean it
            # silently did not apply
            assert not torch.allclose(grad, torch.ones_like(grad))
        else:
            # the paper's estimator copies the gradient through unchanged
            assert torch.allclose(grad, torch.ones_like(grad), atol=1e-5)

        # under either estimator the encoder must actually receive gradient
        assert grad.abs().max() > 0, "no gradient reaches the encoder"

    def test_predict_step_uncertainty_is_non_zero(self, module: Any) -> None:
        """Regression test: stochastic codebook sampling is gated on module.training.

        In eval mode, which is where predict_step runs, every draw would otherwise be
        the same deterministic nearest-neighbour lookup and pred_uct would be
        identically zero without any error being raised.
        """
        module.eval()
        X = torch.randn(2, 3, module.img_size, module.img_size)
        out = module.predict_step(X)

        assert out["pred"].shape == (2, module.out_channels, 64, 64)
        assert out["pred_uct"].shape == out["pred"].shape
        assert not torch.isnan(out["pred_uct"]).any()
        assert out["pred_uct"].abs().sum() > 0

    def test_predict_step_leaves_codebook_untouched(self, module: Any) -> None:
        """Prediction must not drift the codebook via its EMA update.

        With ``kmeans_init=True`` the codebook is all zeros until the first
        forward pass populates it from k-means centroids, so a first
        ``predict_step`` legitimately changes ``embed``. That one-off
        initialization is not the drift this guards against, hence the warm-up
        call before the snapshot.
        """
        module.eval()
        X = torch.randn(2, 3, module.img_size, module.img_size)

        # warm-up: trigger the k-means initialization so the comparison below
        # sees a settled codebook
        module.predict_step(X)
        assert bool(module.vq_module._codebook.initted), (
            "codebook should be initialized after one forward pass"
        )

        before = module.vq_module._codebook.embed.clone()
        was_training = module.vq_module._codebook.training

        module.predict_step(X)

        assert torch.equal(module.vq_module._codebook.embed, before)
        assert module.vq_module._codebook.training == was_training

    def test_single_sample_uncertainty_is_not_nan(self) -> None:
        """The unbiased std of a single draw is nan, so it must be special cased."""
        model_conf = OmegaConf.load(
            "tests/configs/pixelwise_regression/vqvae_conv_encoder.yaml"
        )
        module = instantiate(
            model_conf.uq_method, num_samples=1, sample_codebook_temp=0.0
        )
        module.eval()
        out = module.predict_step(torch.randn(2, 3, module.img_size, module.img_size))

        assert not torch.isnan(out["pred_uct"]).any()
        assert torch.all(out["pred_uct"] == 0)

    def test_deterministic_multi_sample_is_rejected(self) -> None:
        """num_samples > 1 with a zero temperature would give zero uncertainty."""
        model_conf = OmegaConf.load(
            "tests/configs/pixelwise_regression/vqvae_conv_encoder.yaml"
        )
        # hydra wraps the ValueError raised by the constructor
        with pytest.raises(Exception, match="sample_codebook_temp"):
            instantiate(model_conf.uq_method, num_samples=2, sample_codebook_temp=0.0)

    def test_sample_points_at_the_prior(self, module: Any) -> None:
        """A stage one VQ-VAE has no prior over the latent to sample from."""
        with pytest.raises(NotImplementedError, match="VQVAEPrior"):
            module.sample(4)

    def test_outputs_are_contiguous(self, module: Any) -> None:
        """Regression test: torchmetrics calls .view(-1) on the reconstruction.

        With accept_image_fmap=True the codebook lookup permutes the channel axis
        back into place and returns a non-contiguous view, which propagates through
        the decoder and makes that .view(-1) raise at the end of the first epoch.
        """
        X = torch.randn(2, 3, module.img_size, module.img_size)
        quantized, _, _ = module.encode_img_to_latent(X)
        x_recon, _, _ = module.forward(X)

        assert quantized.is_contiguous()
        assert x_recon.is_contiguous()

        module.eval()
        assert module.predict_step(X)["pred"].is_contiguous()

    def test_codebook_metrics_bracket_the_collapse_and_uniform_cases(
        self, module: Any
    ) -> None:
        """Codebook usage is the headline diagnostic, so pin both of its extremes.

        A collapsed codebook is the characteristic VQ-VAE failure and it is
        invisible in the reconstruction loss, so the sweep selects on these two
        numbers. Perplexity is the effective number of codes in use: exactly 1
        when every position picks the same code, exactly ``codebook_size`` when
        the batch spreads uniformly across the whole codebook.
        """
        codebook_size = module.codebook_size

        collapsed = torch.zeros(2, 4, 4, dtype=torch.long)
        perplexity, usage = module.compute_codebook_metrics(collapsed)
        assert torch.isclose(perplexity, torch.tensor(1.0))
        assert torch.isclose(usage, torch.tensor(1.0 / codebook_size))

        uniform = torch.arange(codebook_size).reshape(1, -1).repeat(4, 1).reshape(2, -1)
        perplexity, usage = module.compute_codebook_metrics(uniform)
        assert torch.isclose(perplexity, torch.tensor(float(codebook_size)))
        assert torch.isclose(usage, torch.tensor(1.0))

    def test_plot_and_save_samples_logs_reconstructions(
        self, module: Any, tmp_path: Path, accelerator_config: dict
    ) -> None:
        """The inherited hook draws from a prior this model does not have.

        ``VAE.validation_step`` calls ``plot_and_save_samples`` every
        ``log_samples_every_n_steps``, and the inherited implementation calls
        ``self.sample()``, which a stage-one VQ-VAE raises ``NotImplementedError``
        from. Without the override, validation crashes partway through the first
        epoch rather than at construction time.

        The grid pairs reconstructions against the *target*, not the input: the
        decoder emits ``out_channels``, which this config sets to 1 against a
        3-channel input, and ``make_grid`` cannot stack the two.
        """
        module.log_samples_every_n_steps = 1
        datamodule = ToyPixelwiseRegressionDataModule(
            num_images=2, batch_size=2, image_size=module.img_size
        )
        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))
        trainer.fit(module, datamodule)

        assert list(Path(trainer.default_root_dir).glob("reconstruction_*.png")), (
            "no reconstruction grid was written during validation"
        )

    def test_configure_model_is_idempotent(self, module: Any) -> None:
        """Lightning re-invokes configure_model, which must not rebuild the modules."""
        vq_module, decoder = module.vq_module, module.decoder
        module.configure_model()

        assert module.vq_module is vq_module
        assert module.decoder is decoder


class TestVQVAEPrior:
    @pytest.fixture
    def prior(self) -> Any:
        model_conf = OmegaConf.load(
            "tests/configs/pixelwise_regression/vqvae_prior.yaml"
        )
        return instantiate(model_conf.uq_method)

    def test_vq_vae_is_frozen(self, prior: Any) -> None:
        assert all(param.requires_grad is False for param in prior.vq_vae.parameters())
        assert not prior.vq_vae.training

    def test_optimizer_only_receives_prior_params(self, prior: Any) -> None:
        """Frozen VQ-VAE parameters must not land in the optimizer."""
        optimizer = prior.configure_optimizers()["optimizer"]
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}

        assert optimized == {id(p) for p in prior.pixel_cnn.parameters()}
        assert not any(id(p) in optimized for p in prior.vq_vae.parameters())

    def test_ancestral_sample_shape_and_dtype(self, prior: Any) -> None:
        """The piece the PR was missing: sampling a fresh latent grid and decoding it."""
        samples = prior.sample(num_samples=2)

        img_size = prior.vq_vae.img_size
        assert samples.shape == (2, prior.vq_vae.out_channels, img_size, img_size)
        assert samples.dtype == torch.float32
        assert torch.isfinite(samples).all()

    def test_forward_logits_shape(self, prior: Any) -> None:
        grid = prior.vq_vae.latent_feature_dim
        indices = torch.randint(0, prior.vq_vae.codebook_size, (2, grid, grid))

        logits = prior(indices)

        assert logits.shape == (2, prior.vq_vae.codebook_size, grid, grid)

    def test_train_mode_keeps_the_vq_vae_in_eval(self, prior: Any) -> None:
        """``self.train()`` on the prior must not wake the frozen VQ-VAE.

        ``requires_grad=False`` stops gradients but does nothing about batch norm,
        which updates its running statistics in the forward pass whenever the module
        is in train mode. Lightning calls ``train()`` on the whole module at the top
        of every training epoch, so without the override each epoch would quietly
        shift the encoder's normalization -- changing the latents the prior is being
        fit against, and the reconstructions of an already-trained VQ-VAE, while
        every parameter stayed bit-identical.
        """
        prior.train()

        assert prior.training
        assert prior.pixel_cnn.training
        assert not prior.vq_vae.training

        batch_norms = [
            m for m in prior.vq_vae.modules() if isinstance(m, torch.nn.BatchNorm2d)
        ]
        assert batch_norms, "expected the resnet encoder to contain batch norm"
        before = [bn.running_mean.clone() for bn in batch_norms]

        prior.vq_vae.encode_img_to_latent(
            torch.randn(2, 3, prior.vq_vae.img_size, prior.vq_vae.img_size)
        )

        for bn, running_mean in zip(batch_norms, before):
            assert torch.equal(bn.running_mean, running_mean), (
                "batch norm statistics moved while the prior was in train mode"
            )

    def test_fit_leaves_vq_vae_bit_identical(
        self, prior: Any, tmp_path: Path, accelerator_config: dict
    ) -> None:
        """Lightning re-invokes configure_model at fit, which must not reset weights."""
        before = {
            name: param.clone() for name, param in prior.vq_vae.named_parameters()
        }

        datamodule = ToyPixelwiseRegressionDataModule(
            num_images=2, batch_size=2, image_size=prior.vq_vae.img_size
        )
        trainer = Trainer(**minimal_trainer_kwargs(accelerator_config, tmp_path))
        trainer.fit(prior, datamodule)

        after = dict(prior.vq_vae.named_parameters())
        assert set(after) == set(before)
        for name, param in before.items():
            assert torch.equal(after[name].cpu(), param.cpu()), (
                f"VQ-VAE parameter {name} changed during prior training"
            )


class TestPixelCNN:
    """Tests for the autoregressive prior network."""

    def test_receptive_field_is_causal_in_raster_scan_order(self) -> None:
        """The output at a position must see every earlier position and no other.

        This is the one property that makes ancestral sampling valid, and a mask
        that is wrong in either direction fails silently. If the current position
        leaks in, the model trains to copy its own input: cross entropy collapses
        towards zero and accuracy towards one, which looks like an excellent prior
        while ``sample()`` -- which only ever sees positions it has already filled --
        produces noise. If an earlier position is wrongly masked out, the prior is
        merely weaker, with nothing to distinguish it from underfitting.

        The dependency structure is probed directly: perturb one input index at a
        time and check whether the logits at a fixed position move.
        """
        grid, codebook_size = 6, 8
        row, col = 3, 3

        pixel_cnn = PixelCNN(num_embeddings=codebook_size, c_hidden=8).eval()
        indices = torch.randint(0, codebook_size, (1, grid, grid))

        with torch.no_grad():
            reference = pixel_cnn(indices)[0, :, row, col]

            for probe_row in range(grid):
                for probe_col in range(grid):
                    perturbed = indices.clone()
                    perturbed[0, probe_row, probe_col] = (
                        perturbed[0, probe_row, probe_col] + 1
                    ) % codebook_size
                    logits = pixel_cnn(perturbed)[0, :, row, col]
                    depends = not torch.allclose(logits, reference, atol=1e-6)

                    is_earlier = (probe_row, probe_col) < (row, col)
                    assert depends == is_earlier, (
                        f"logits at ({row}, {col}) "
                        f"{'depend on' if depends else 'ignore'} ({probe_row}, "
                        f"{probe_col}), which comes "
                        f"{'earlier' if is_earlier else 'at or later'} in raster-scan "
                        "order"
                    )

    def test_forward_shape_matches_cross_entropy_layout(self) -> None:
        """Logits are consumed directly by ``F.cross_entropy`` against the indices."""
        pixel_cnn = PixelCNN(num_embeddings=8, c_hidden=8)
        indices = torch.randint(0, 8, (2, 4, 4))

        logits = pixel_cnn(indices)

        assert logits.shape == (2, 8, 4, 4)
        torch.nn.functional.cross_entropy(logits, indices).backward()

    def test_masked_weights_stay_zero_after_an_update(self) -> None:
        """The mask is applied to ``weight.data``, not enforced by construction.

        ``MaskedConvolution.forward`` multiplies the weights in place each call, so
        an optimizer step can write non-zero values into masked positions between
        calls. This pins that the next forward pass clears them again, which is what
        keeps causality intact across training rather than only at initialization.
        """
        pixel_cnn = PixelCNN(num_embeddings=8, c_hidden=8)
        conv = pixel_cnn.conv_vstack

        with torch.no_grad():
            conv.conv.weight.fill_(1.0)
        pixel_cnn(torch.randint(0, 8, (1, 4, 4)))

        masked = conv.conv.weight[:, :, conv.mask[0, 0] == 0]
        assert masked.numel() > 0, "expected the vertical stack to mask some weights"
        assert torch.all(masked == 0)


class TestVQVAELoss:
    def test_commit_scale_weights_only_the_commitment_term(self) -> None:
        """Beta scales the commitment loss and must leave reconstruction alone.

        The two terms are returned separately so they can be logged separately, and
        the sweep varies beta, so a scale applied to the wrong term would move both
        the objective and the metric used to compare runs.
        """
        x_recon = torch.zeros(2, 1, 4, 4)
        target = torch.ones(2, 1, 4, 4)
        commit_loss = torch.tensor(2.0)

        scaled_commit, recon = VQVAELoss(commit_scale=0.25)(
            x_recon, target, commit_loss
        )

        assert torch.isclose(scaled_commit, torch.tensor(0.5))
        assert torch.isclose(recon, torch.tensor(1.0))

        # doubling beta doubles the commitment term and nothing else
        doubled_commit, doubled_recon = VQVAELoss(commit_scale=0.5)(
            x_recon, target, commit_loss
        )
        assert torch.isclose(doubled_commit, 2 * scaled_commit)
        assert torch.isclose(doubled_recon, recon)
