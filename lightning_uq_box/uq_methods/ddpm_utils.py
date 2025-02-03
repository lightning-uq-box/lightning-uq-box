# Uses https://github.com/lucidrains/denoising-diffusion-pytorch/
# MIT License
# Copyright (c) 2020 Phil Wang

# adapted denoising-diffusion-pytorch to be a lightning module
# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

import torch
import math


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Generate a linear beta schedule for the given number of timesteps.

    Args:
        timesteps: The number of timesteps for the schedule.

    Returns:
         A tensor containing the beta values for each timestep.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Generate a cosine beta schedule for the given number of timesteps.

    Args:
        timesteps: The number of timesteps for the schedule.
        s: A small constant to prevent division by zero. Defaults to 0.008.

    Returns:
        A tensor containing the beta values for each timestep.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    """
    Generate a sigmoid beta schedule for the given number of timesteps.

    Args:
        timesteps: The number of timesteps for the schedule.
        start: The starting value for the sigmoid function. Defaults to -3.
        end: The ending value for the sigmoid function. Defaults to 3.
        tau: The temperature parameter for the sigmoid function. Defaults to 1.
        clamp_min: The minimum value to clamp the betas to. Defaults to 1e-5.

    Returns:
        A tensor containing the beta values for each timestep.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def quadratic_beta_schedule(timesteps):
    if timesteps < 20:
        raise ValueError(
            "Warning: Less than 20 timesteps require adjustments to this schedule!"
        )
    beta_start = 0.0001 * (
        1000 / timesteps
    )  # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000 / timesteps)
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    return torch.clip(betas, 0.0001, 0.9999)


def cubic_beta_schedule(timesteps):
    if timesteps < 20:
        raise ValueError(
            "Warning: Less than 20 timesteps require adjustments to this schedule!"
        )
    beta_start = 0.0001 * (
        1000 / timesteps
    )  # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000 / timesteps)
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 3
    return 2 * torch.clip(betas, 0.0001, 0.9999)


def exponential_beta_schedule(
    timesteps,
    min_noise_level,
    etas_end: float = 0.99,
    kappa: float = 1.0,
    power: float = 1.0,
):
    """Exponential beta schedule."""
    # https://github.com/zsyOAOA/ResShift/blob/dfc2ff705a962de1601a491511b43a93b97d9622/models/gaussian_diffusion.py#L45
    betas_start = min(min_noise_level / kappa, min_noise_level)
    increaser = math.exp(1 / (timesteps - 1) * math.log(etas_end / betas_start))
    base = torch.ones(timesteps) * increaser
    power_timestep = torch.linspace(0, 1, timesteps) ** power
    power_timestep *= timesteps - 1
    sqrt_betas = torch.pow(base, power_timestep) * betas_start
    return torch.pow(sqrt_betas, 2)
