<p align="center">
<img src="https://github.com/lightning-uq-box/lightning-uq-box/blob/main/docs/_static/lettering.jpeg?raw=true" alt="Lightning-UQ-Box logo" width="600" height="auto" />
</p>

[![docs](https://readthedocs.org/projects/lightning-uq-box/badge/?version=latest)](https://lightning-uq-box.readthedocs.io/en/latest/)
[![style](https://github.com/lightning-uq-box/lightning-uq-box/actions/workflows/style.yaml/badge.svg)](https://github.com/lightning-uq-box/lightning-uq-box/actions/workflows/style.yaml)
[![tests](https://github.com/lightning-uq-box/lightning-uq-box/actions/workflows/tests.yaml/badge.svg)](https://github.com/lightning-uq-box/lightning-uq-box/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/lightning-uq-box/lightning-uq-box/branch/main/graph/badge.svg?token=oa3Z3PMVOg)](https://app.codecov.io/gh/lightning-uq-box/lightning-uq-box)

# lightning-uq-box

The lightning-uq-box is a PyTorch library that provides various Uncertainty Quantification (UQ) techniques for modern neural network architectures.

We hope to provide the starting point for a collaborative open source effort to make it easier for practitioners to include UQ in their workflows and
remove possible barriers of entry. Additionally, we hope this can be a pathway to more easily compare methods across UQ frameworks and potentially enhance the development of new UQ methods for neural networks.

*The project is currently under active development, but we nevertheless hope for early feedback, feature requests, or contributions. Please check the [Contribution Guide](https://lightning-uq-box.readthedocs.io/en/latest/contribute.html) for further information.*

The goal of this library is threefold:

1. Provide implementations for a variety of Uncertainty Quantification methods for Modern Deep Neural Networks that work with a range of neural network architectures and have different theoretical underpinnings
2. Make it easy to compare UQ methods on a given dataset
3. Focus on reproducibility of experiments with minimum boiler plate code and standardized evaluation protocols

To this end, each UQ-Method is essentially just a [Lightning Module](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) which can be used with a [Lightning Data Module](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) and a [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) to execute training, evaluation and inference for your desired task. The library also utilizes the [Lightning Command Line Interface (CLI)](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) for better reproducibility of experiments and setting up experiments at scale.

# Installation

```console
$ pip install lightning-uq-box
```

# UQ-Methods

In the tables that follow below, you can see what UQ-Method/Task combination is currently supported by the Lightning-UQ-Box via these indicators:

- ✅ supported
- ❌ not designed for this task
- ⏳ in progress

The implemented methods are of course not exhaustive, as the number of new methods keeps increasing. For an overview of methods that we are tracking or are planning to support, take a look at [this issue](https://github.com/lightning-uq-box/lightning-uq-box/issues/43).

## Classification of UQ-Methods

The following sections aims to give an overview of different UQ-Methods by grouping them according to some commonalities. We agree that there could be other groupings as well and welcome suggestions to improve this overview. We also follow this grouping for the API documentation in the hopes to make navigation easier.

### Single Forward Pass Methods

| UQ-Method                                     | Regression | Classification | Segmentation | Pixel Wise Regression |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| Quantile Regression (QR)                      |     ✅     |       ❌       |      ❌      |          ✅           |
| Deep Evidential (DE)                          |     ✅     |       ⏳       |      ⏳      |          ✅           |
| Mean Variance Estimation (MVE)                |     ✅     |       ❌       |      ❌      |          ✅           |

### Approximate Bayesian Methods

| UQ-Method                                     | Regression | Classification | Segmentation | Pixel Wise Regression |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| Bayesian Neural Network VI ELBO (BNN_VI_ELBO) |     ✅     |       ✅       |      ✅      |          ⏳           |
| Bayesian Neural Network VI (BNN_VI)           |     ✅     |       ⏳       |      ⏳      |          ⏳           |
| Deep Kernel Learning (DKL)                    |     ✅     |       ✅       |      ❌      |          ❌           |
| Deterministic Uncertainty Estimation (DUE)    |     ✅     |       ✅       |      ❌      |          ❌           |
| Laplace Approximation (Laplace)               |     ✅     |       ✅       |      ❌      |          ❌           |
| Monte Carlo Dropout (MC-Dropout)              |     ✅     |       ✅       |      ✅      |          ✅           |
| Stochastic Gradient Langevin Dynamics (SGLD)  |     ✅     |       ✅       |      ⏳      |          ⏳           |
| Spectral Normalized Gaussian Process (SNGP)   |     ✅     |       ✅       |      ❌      |          ❌           |
| Stochastic Weight Averaging Gaussian (SWAG)   |     ✅     |       ✅       |      ✅      |          ✅           |
| Variational Bayesian Last Layer (VBLL)        |     ✅     |       ✅       |      ❌      |          ❌           |
| Deep Ensemble                                 |     ✅     |       ✅       |      ✅      |          ✅           |
| Masked Ensemble                               |     ✅     |       ✅       |      ⏳      |          ⏳           |

### Generative Models

| UQ-Method                                     | Regression | Classification | Segmentation | Pixel Wise Regression |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| Classification And Regression Diffusion (CARD)|     ✅     |       ✅       |      ❌      |          ❌           |
| Probabilistic UNet                            |     ❌     |       ❌       |      ✅      |          ❌           |
| Hierarchical Probabilistic UNet               |     ❌     |       ❌       |      ✅      |          ❌           |

### Post-Hoc methods

| UQ-Method                                     | Regression | Classification | Segmentation | Pixel Wise Regression |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| Test Time Augmentation (TTA)                  |     ✅     |       ✅       |      ⏳      |          ⏳           |
| Temperature Scaling                           |     ❌     |       ✅       |      ⏳      |          ❌           |
| Conformal Quantile Regression (Conformal QR)  |     ✅     |       ❌       |      ❌      |          ⏳           |
| Regularized Adaptive Prediction Sets (RAPS)   |     ❌     |       ✅       |      ❌      |          ❌           |
| Image to Image Conformal                      |     ❌     |       ❌       |      ❌      |          ✅           |

# Tutorials

We try to provide many different tutorials so that users can get a better understanding of implemented methods and get a feel for how they apply to different problems.
Head over to the [tutorials](https://lightning-uq-box.readthedocs.io/en/latest/tutorial_overview.html) page to get started. These tutorials can also be launched in google colab if you navigate to the rocket icon at the top of a tutorial page.

# Documentation
We aim to provide an extensive documentation on all included UQ-methods that provide some theoretical background, as well as tutorials that illustrate these methods on toy datasets.
