# Lightning-UQ-Box

Welcome to the lightning-uq-box documentation page. Our goal is to give you both an intuition of how different UQ-Methods work as well as demonstrate how you can apply these methods in your research or projects. To this end, we aim to give both a theoretical and practical overview of implemented UQ-Methods since there exist a wide variety of UQ-Methods. Similarly, there are several general tasks for which practitioners might require uncertainty estimates. The library currently supports the following four tasks:

1. **Regression** for tabular/image inputs with 1D scalar targets
2. 2D Regression / **Pixel Wise Regression**
3. **Classification** for tabular/image inputs with single classification label
4. **Segmentation** where each pixel is assigned a class

While some UQ-Methods like MC-Dropout or Deep Ensembles can be applied across tasks, other methods are specifically developed for certain tasks. The following aims to give an overview of supported methods for the different tasks.

In the tables that follow below, you can see what UQ-Method/Task combination is currently supported by the Lightning-UQ-Box via these indicators:

- ✅ supported
- ❌ not designed for this task
- ⏳ in progress

To get started, 

```console
$ pip install lightning-uq-box
```

## Classification of UQ-Methods

The following sections aims to give an overview of different UQ-Methods by grouping them according to some commonalities. We agree that there could be other groupings as well and welcome suggestions to improve this overview. We also follow this grouping for the API documentation in the hopes to make navigation easier.

### Single Forward Pass Methods

| UQ-Method                                     | Regression | Classification | Segmentation | Pixel Wise Regression |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| Quantile Regression (QR)                      |     ✅     |       ❌       |      ❌      |          ⏳           |
| Deep Evidential (DE)                          |     ✅     |       ⏳       |      ⏳      |          ⏳           |
| Mean Variance Estimation (MVE)                |     ✅     |       ❌       |      ❌      |          ⏳           |

### Approximate Bayesian Methods

| UQ-Method                                     | Regression | Classification | Segmentation | Pixel Wise Regression |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| Bayesian Neural Network VI ELBO (BNN_VI_ELBO) |     ✅     |       ✅       |      ✅      |          ⏳           |
| Bayesian Neural Network VI (BNN_VI)           |     ✅     |       ⏳       |      ⏳      |          ⏳           |
| Deep Kernel Learning (DKL)                    |     ✅     |       ✅       |      ❌      |          ❌           |
| Deterministic Uncertainty Estimation (DUE)    |     ✅     |       ✅       |      ❌      |          ❌           |
| Laplace Approximation (Laplace)               |     ✅     |       ✅       |      ❌      |          ❌           |
| Monte Carlo Dropout (MC-Dropout)              |     ✅     |       ✅       |      ✅      |          ⏳           |
| Stochastic Gradient Langevin Dynamics (SGLD)  |     ✅     |       ✅       |      ⏳      |          ⏳           |
| Spectral Normalized Gaussian Process (SNGP)   |     ✅     |       ✅       |      ❌      |          ❌           |
| Stochastic Weight Averaging Gaussian (SWAG)   |     ✅     |       ✅       |      ✅      |          ⏳           |
| Deep Ensemble                                 |     ✅     |       ✅       |      ✅      |          ⏳           |

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


## Table of contents

```{toctree}
:maxdepth: 2

user_guide
tutorial_overview
running_experiments
api/index
contribute
GitHub Repository <https://github.com/lightning-uq-box/lightning-uq-box>
```
