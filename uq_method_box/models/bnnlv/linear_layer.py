"""Linear Variational Layer adapted for Alpha Divergence."""

import math

import torch
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
from torch import Tensor


class LinearFlipoutLayer(LinearFlipout):
    """Linear Flipout Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
        bias: bool = True,
    ) -> None:
        """Initialize a new Linear Flipout layer.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            prior_mean: mean of the prior arbitrary distribution to be used on the
                complexity cost
            prior_variance: variance of the prior arbitrary distribution to be used
                on the complexity cost
            posterior_mu_init: init trainable mu parameter representing mean of the
                approximate posterior
            posterior_rho_init: init trainable rho parameter representing the sigma
                of the approximate posterior through softplus function
            bias: if set to False, the layer will not learn an additive bias.
        """
        super().__init__(
            in_features,
            out_features,
            prior_mean,
            prior_variance,
            posterior_mu_init,
            posterior_rho_init,
            bias,
        )

        self.weight_eps = None

    def sample_weights(self, n_samples: int) -> tuple[Tensor]:
        """Sample variational weights.

        Args:
            n_samples: number of samples to return

        Returns:
            weight and bias sample for layer of shape [num_samples, num_parameters]
        """
        if self.weight_eps is None:
            weights_eps = torch.randn(
                n_samples, self.out_features, self.in_features
            ).to(self.mu_weight.device)
            bias_eps = torch.randn(n_samples, self.out_features).to(
                self.mu_weight.device
            )

        else:
            weights_eps = self.weight_eps[:n_samples].to(self.mu_weight.device)
            bias_eps = self.bias_eps[:n_samples].to(self.mu_weight.device)

        # ensure sigma weight and bias are positive
        weight_sigma = F.softplus(self.rho_weight)
        bias_sigma = F.softplus(self.rho_bias)

        # sample weight and bias with reparameterization trick
        weight_sample = self.mu_weight + weights_eps * weight_sigma
        bias_sample = self.mu_bias + bias_eps * bias_sigma

        all_params_mu = torch.cat([self.mu_weight.flatten(), self.mu_bias.flatten()])
        all_params_sigma = torch.cat([weight_sigma.flatten(), bias_sigma.flatten()])
        all_sample = torch.cat([weight_sample.flatten(1), bias_sample.flatten(1)], 1)

        self.log_normalizer = self.calc_log_normalizer(all_params_mu, all_params_sigma)
        self.log_f_hat = self.calc_log_f_hat(
            all_sample, all_params_mu, all_params_sigma
        )
        return weight_sample, bias_sample

    # TODO think about most convenient way to introduce n_samples to maybe have control
    # over
    def forward(self, input: Tensor, n_samples: int = 25) -> Tensor:
        """Forward pass through linear layer.

        Args:
            input: input tensor to linear layer
            n_samples: how many samples to draw

        Returns:
            computed output of shape [n_samples, batch_size, out_features]
        """
        inp_shape = input.shape
        if len(inp_shape) == 2:
            input = input.unsqueeze(0).repeat(n_samples, 1, 1)
        else:
            n_samples = inp_shape[0]

        weight, bias = self.sample_weights(n_samples)

        state = input.matmul(weight.transpose(-1, -2)) + bias.unsqueeze(-1).transpose(
            1, 2
        )  # n_samples x batch_size x out_features

        return state

    def calc_log_Z_prior(self) -> Tensor:
        """Compute log Z prior.

        Returns:
            tensor of shape 0
        """
        n_params = self.mu_weight.numel() + self.mu_bias.numel()
        return torch.tensor(
            0.5 * n_params * math.log(self.prior_variance * 2 * math.pi)
        )

    def calc_log_f_hat(self, w: Tensor, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute equation 3.16.

        Args:
            w: sampled weight matrix [n_samples, num_params]
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            summed log f hat over the parameters [n_samples]
        """
        v_W = std_W**2
        m_W = m_W
        # natural parameters: -1/(2 sigma^2), mu/(sigma^2)
        # \lambda is (\lambda_q - \lambda_prior) / N
        # assuming prior mean is 0 and moving N calculation outside
        return (
            ((v_W - self.prior_variance) / (2 * self.prior_variance * v_W)) * (w**2)
            + (m_W / v_W) * w
        ).sum(axis=1)

    def calc_log_normalizer(self, m_W: Tensor, std_W: Tensor) -> Tensor:
        """Compute left summand of 3.18.

        Args:
            m_W: mean weight matrix at current iteration [num_params]
            std_W: sigma weight matrix at current iteration [num_params]

        Returns:
            log normalizer summed over all layer parameters shape 0
        """
        v_W = std_W**2
        m_W = m_W
        # return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()
        return (0.5 * torch.log(v_W * 2 * math.pi) + 0.5 * m_W**2 / v_W).sum()


# if __name__ == "__main__":
#     layer = LinearFlipoutLayer(in_features=1, out_features=10)
#     x = torch.randn(size=(32, 1))
#     state = layer(x)
#     layer_loss_dict = {
#         "log_Z_prior": layer.calc_log_Z_prior(),
#         "log_f_hat": layer.log_f_hat,
#         "log_normalizer": layer.log_normalizer,
#     }
#     print(f"state_shape: {state.shape}", layer_loss_dict)
