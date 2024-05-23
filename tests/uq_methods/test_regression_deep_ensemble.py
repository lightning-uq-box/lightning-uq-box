import os
from functools import partial

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from lightning_uq_box.datamodules.utils import collate_fn_tensordataset
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import DeepEnsembleRegression, MVERegression


def gt_func(x):
    """Ground truth function. Just a sine, as in
    uncertainty_toolbox.data.synthetic_sine_heteroscedastic()."""
    return np.sin(x)


def synthetic_heteroscedastic(
    n_points: int = 10, func=gt_func, bounds=[0, 15], rng=None
):
    """Adapted from uncertainty_toolbox.data.synthetic_sine_heteroscedastic()."""
    assert rng is not None
    x = np.linspace(bounds[0], bounds[1], n_points)
    std = 0.01 + np.abs(x - 5.0) / 10.0
    return x, gt_func(x) + rng.normal(scale=std)


class Data1D:
    def __init__(
        self,
        n_points: int = 200,
        batch_size: int = 100,
        test_fraction: float = 0.2,
        val_fraction: float = 0.1,
        noise_seed: int = 42,
        split_seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(noise_seed)
        self.batch_size = batch_size

        x, y = synthetic_heteroscedastic(n_points, rng=rng)

        # train + test + val data
        self.X_all = x[:, None]
        self.Y_all = y[:, None]

        X_other, self.X_test, Y_other, self.Y_test = train_test_split(
            self.X_all, self.Y_all, test_size=test_fraction, random_state=split_seed
        )

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            X_other,
            Y_other,
            test_size=val_fraction / (1 - test_fraction),
            random_state=split_seed,
        )

        # Fit scalers on train data
        scalers = dict(
            X=StandardScaler().fit(self.X_train), Y=StandardScaler().fit(self.Y_train)
        )

        # Apply scaling to all splits, convert to torch tensors
        for xy in ["X", "Y"]:
            for arr_type in ["train", "test", "val", "all"]:
                arr_name = f"{xy}_{arr_type}"
                setattr(
                    self,
                    arr_name,
                    self._n2t(scalers[xy].transform(getattr(self, arr_name))),
                )

    @staticmethod
    def _n2t(x):
        return torch.from_numpy(x).type(torch.float32)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.Y_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tensordataset,
        )


def test_mve_gmm_single_model(tmp_path):
    """Test whether DeepEnsembleRegression reduces to a single MVE model when
    n_ensemble_members=1.
    """
    seed_everything(123)

    data = Data1D(n_points=100, batch_size=100)
    step_size = 1e-2
    max_epochs = 3
    burnin_epochs = 0
    n_hidden = [20, 20]

    optimizer = partial(
        torch.optim.LBFGS, lr=step_size, line_search_fn="strong_wolfe", max_iter=10
    )

    mlp_model = MLP(n_hidden=n_hidden, n_outputs=2, activation_fn=torch.nn.Tanh())
    ensemble_member = MVERegression(
        mlp_model, optimizer=optimizer, burnin_epochs=burnin_epochs
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(ensemble_member, train_dataloaders=data.train_dataloader())
    save_path = os.path.join(tmp_path, "model.ckpt")
    trainer.save_checkpoint(save_path)
    trained_models = [dict(base_model=ensemble_member, ckpt_path=save_path)]

    pred_model_mve = ensemble_member
    pred_model_mve_gmm = DeepEnsembleRegression(
        n_ensemble_members=len(trained_models), ensemble_members=trained_models
    )

    pred_mve = pred_model_mve.predict_step(data.X_test)
    pred_mve_gmm = pred_model_mve_gmm.predict_step(data.X_test)

    # pred_uct, aleatoric_uct, pred
    for key in set(pred_mve.keys()) & set(pred_mve_gmm.keys()):
        aa = pred_mve[key].squeeze()
        bb = pred_mve_gmm[key].squeeze()
        # NOTE: max(abs(aa-bb)) ~ 1.2e-07, that's quite high, even for float32.
        # Either one of the hard-cocded eps values, or too much of
        # sqrt(exp(log(...))**2).
        assert torch.allclose(
            aa, bb, rtol=0, atol=1e-6
        ), f"{torch.max(torch.abs(aa-bb))=}"
