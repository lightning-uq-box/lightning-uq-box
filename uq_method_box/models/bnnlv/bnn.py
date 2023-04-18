import torch.nn as nn
import torch
import torch.nn.functional as F
from dense_variational import DenseVariational, LV
from torch.distributions import Normal


class BNN(nn.Module):
    """DQN Model with BNN Linear layers"""

    DEFAULT_CONFIG = {
        "n_samples": 25,
        "device": None,
        "log_aleatoric_std_init": -2.5,
        "graph": [20, 20],
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        config={},
    ):
        super(BNN, self).__init__()
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.device = self.config["device"]

        self.net = nn.ModuleList(
            [DenseVariational(input_dim, self.config["graph"][0], self.config)]
            + [
                DenseVariational(
                    self.config["graph"][i], self.config["graph"][i + 1], self.config
                )
                for i in range(len(self.config["graph"]) - 1)
            ]
            + [DenseVariational(self.config["graph"][-1], output_dim, self.config)]
        )

        self.log_aleatoric_std = nn.Parameter(
            torch.tensor(
                [self.config["log_aleatoric_std_init"] for _ in range(output_dim)],
                device=self.device,
            )
        )

    def forward(self, x):
        """Forward pass of the BDQN"""
        x = F.relu(self.net[0](x))
        for i in range(1, len(self.net) - 1):
            x = F.relu(self.net[i](x))
        x = self.net[-1](x)
        ll = Normal(x.transpose(0, 1), torch.exp(self.log_aleatoric_std))
        return ll

    def get_loss_terms(self):
        return {
            "log_Z_prior": self.log_Z_prior(),
            "log_f_hat": self.log_f_hat(),
            "log_normalizer": self.log_normalizer(),
            "log_f_hat_z": 0.0,
            "log_normalizer_z": 0.0,
        }

    def log_Z_prior(self):
        """Log Z prior of the weights"""
        log_Z_prior = torch.stack([layer.log_Z_prior for layer in self.net])
        return torch.sum(log_Z_prior, 0)

    def log_f_hat(self):
        """Log f hat of the weights"""
        log_f_hat = torch.stack([layer.log_f_hat for layer in self.net])
        return torch.sum(log_f_hat, 0)

    def log_normalizer(self):
        """Log normalizer of the weights"""
        log_normalizer = torch.stack([layer.log_normalizer for layer in self.net])
        return torch.sum(log_normalizer, 0)


class BNNLV(BNN):
    DEFAULT_CONFIG = {
        "n_samples": 25,
        "device": None,
        "log_aleatoric_std_init": -2.5,
        "graph": [20, 20],
        "lv_latent_dim": 1,
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        N,
        config={},
    ):
        self.config = {**self.DEFAULT_CONFIG, **config}

        super(BNNLV, self).__init__(input_dim + 1, output_dim, config)

        self.lv = LV(N, self.config)

    def forward(self, x_in, training=True):
        if len(x_in) == 2:
            x, z_ind = x_in
            if training:
                z = self.lv(z_ind)  # [S,N,latent_dim]
            else:
                z = z_ind
        else:
            x = x_in
            z = torch.randn(
                self.config["n_samples"], x.shape[0], self.config["lv_latent_dim"]
            ).to(self.device)

        # tile x to [S,N,D]
        if len(x.shape) == 2:
            x = x[None, :, :].repeat(z.shape[0], 1, 1)
        x = torch.cat([x, z], -1)
        return super(BNNLV, self).forward(x)

    def get_loss_terms(self):
        return {
            "log_Z_prior": self.log_Z_prior(),
            "log_f_hat": self.log_f_hat(),
            "log_normalizer": self.log_normalizer(),
            "log_f_hat_z": self.lv.log_f_hat_z,
            "log_normalizer_z": self.lv.log_normalizer_z,
        }
