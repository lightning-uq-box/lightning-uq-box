"""CARDS Model Utilities."""

import torch
from torch import Tensor
import torch.nn as nn
import math


class NoiseScheduler:
    """Noise Scheduler for Diffusion Training."""
    valid_schedules = ["linear", "const", "quad", "jsd", "sigmoid", "cosine", "cosine_anneal"]

    def __init__(self, schedule: str="linear", n_steps: int = 1000, beta_start: float=1e-5, beta_end: float=1e-2) -> None:
        """Initialize a new instance of the noise scheduler.
        
        Args:
            schedule: 
            n_steps: number of diffusion time steps
            beta_start: beta noise start value
            beta_end: beta noise end value
        Raises:
            AssertionError if schedule is invalid
        """
        assert schedule in self.valid_schedules, f"Invalid schedule, please choose one of {self.valid_schedules}."
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
            "cosine_anneal": self.cosine_anneal_schedule()
        }[schedule]

        self.betas_sqrt = torch.sqrt(self.betas)
        self.alphas = 1.0 -self.betas
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
        return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.n_steps) ** 2

    def sigmoid_schedule(self) -> Tensor:
        """Sigmoid Schedule."""
        betas = torch.sigmoid(torch.linspace(-6, 6, self.n_steps)) * (self.beta_end - self.beta_start) + self.beta_start
        return torch.sigmoid(betas) 

    def cosine_schedule(self) -> Tensor:
        """Cosine Schedule."""
        max_beta = 0.999
        cosine_s = 0.008
        return torch.tensor(
            [min(1 - (math.cos(((i + 1) / self.n_steps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / self.n_steps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(self.n_steps)])
        
    def cosine_anneal_schedule(self) -> Tensor:
        """Cosine Annealing Schedule."""
        return torch.tensor(
            [self.beta_start + 0.5 * (self.beta_end - self.beta_start) * (1 - math.cos(t / (self.n_steps - 1) * math.pi)) for t in
             range(self.n_steps)])
    

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
    

class ConditionalLinear(nn.Module):
    """Conditional Linear Layer."""
    def __init__(self, n_inputs: int, n_outputs: int, n_steps: int) -> None:
        """Initialize a new instance of the layer.
        
        Args:
            n_inputs: number of inputs to the layer
            n_outputs: number of outputs from the layer
            n_steps: number of diffusion steps in embedding
        
        """
        super(ConditionalLinear, self).__init__()
        self.n_outputs = n_outputs
        self.lin = nn.Linear(n_inputs, n_outputs)
        self.embed = nn.Embedding(n_steps, n_outputs)
        self.embed.weight.data.uniform_()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass of conditional linear layer.
        
        Args:
            x: input of shape [N, n_inputs]
            t: input of shape [1]

        Returns:
            output from condtitional linear model of shape [N, n_outputs]
        """
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.n_outputs) * out
        return out
    
    # def extra_repr(self) -> str:
    #     """Representation when printing out Layer."""
    #     return "in_features={}, out_features={}, ".format(
    #         self.n_inputs, self.out_features, self.bias is not None, self.is_frozen
    #     )


class DiffusionSequential(nn.Sequential):
    """My Sequential to accept multiple inputs."""
    def forward(self, input: Tensor, t: Tensor):
        """Forward pass.
        
        Args:
            input: input tensor to model shape [n, feature_dim]
            t: time steps shape [1]

        Returns:
            output of diffusion model [n, output_dim]
        """
        for module in self._modules.values():
            if isinstance(module, ConditionalLinear):
                input = module(input, t)
            else:
                input = module(input)
        return input   

class ConditionalGuidedLinearModel(nn.Module):
    """Conditional Guided Model."""
    def __init__(self, n_steps: int, x_dim: int, y_dim: int, z_dim: int, n_hidden: list[int] = [64, 64], n_outputs: int = 1, cat_x: bool = False, cat_y_pred: bool = False, activation_fn: nn.Module=nn.Softplus()) -> None:
        """Initialize a new instance of Conditional Guided Model.
        
        Args:
            n_steps:
            x_dim:
            y_dim:
            z_dim
            n_hidden: number of Conditional Linear Layers with dimension
            n_outputs: 
            cat_x:
            cat_y_pred:
            activation_fn: activation function between conditional linear layers
        """
        super(ConditionalGuidedLinearModel, self).__init__()
       
        self.cat_x = cat_x
        self.cat_y_pred = cat_y_pred
        data_dim = y_dim
        if self.cat_x:
            data_dim += x_dim
        if self.cat_y_pred:
            data_dim += y_dim
        layer_sizes = [data_dim] + n_hidden
        layers = []
        for idx in range(1, len(layer_sizes)):
            layers += [
                ConditionalLinear(layer_sizes[idx-1], layer_sizes[idx], n_steps),
                activation_fn
            ]
        # final output layer is standard layer
        layers += [nn.Linear(layer_sizes[-1], n_outputs)]
        self.model = DiffusionSequential(*layers)

    def forward(self, x: Tensor, y_t: Tensor, y_0_hat: Tensor, t: int) -> Tensor:
        """Forward pass of the Conditional Guided Model.
        
        Args:
            x: 
            y:
            y_0_hat:
            t: time step
        """
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        
        return self.model(eps_pred, t)