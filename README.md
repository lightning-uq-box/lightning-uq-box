<p align="center">
<img src="docs/_static/lettering.png" alt="Lightning-UQ-Box logo" width="500" height="150" />
</p>

# lightning-uq-box

The lightning-uq-box is a PyTorch library that provides various Uncertainty Quantification (UQ) techniques for modern neural network architectures. 

We hope to provide the starting point for a collobrative open source effort to make it easier for practicioners to include UQ in their workflows and
remove possible barriers of entry. Additionally, we hope this can be a pathway to more easily compare methods across UQ frameworks and potentially enhance the development of new UQ methods for neural networks.

*The project is currently under active development, but we nevertheless hope for early feedback, feature requests, or contributions. Please check the "Contribution" section for further information.*

The goal of this library is threefold:

1. Provde access to a variety of Uncertainty Quantification methods for Modern Deep Neural Networks that work with a range of neural network architectures.
2. Make it easy to compare methods on a new dataset or a new method on existing datasets.
3. Focus on reproducibility of experiments with miminum boiler plate code and standardized evaluation protocols.

To this end, each UQ-Method is essentially nothing more than a [Lightning Module](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) which can be used with [Lightning Data Module](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) and a [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) to exectute training, evaluation and inference for your desired task. The library also utilizes the [Lightning Command Line Interface (CLI)](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) for better reproducability of experiments and setting up experiments at scale.


# UQ-Methods

In the tables that follow below, you can see what UQ-Method/Task combination is currently supported by the Lightning-UQ-Box via these indicators:

- ✅ supported
- ❌ not designed for this task
- ⏳ in progress

## Classification of UQ-Methods

The following sections aims to give an overview of different UQ-Methods by grouping them according to some commonalities. We agree that there could be other groupings as well and welcome suggestions to improve this overview. We also follow this grouping for the API documentation in the hopes to make navigation easier.

### Single Forward Pass Methods

| UQ-Method            | Regression            | Classification            | Segmentation              | Pixel Wise Regression      |
|----------------------|:---------------------:|:-------------------------:|:-------------------------:|:--------------------------:|
| Quantile Regression  |          ✅           |           ❌              |           ❌              |            ⏳            |
| Deep Evidential      |          ✅           |           ⏳              |           ⏳              |            ⏳            |
| MVE                  |          ✅           |           ❌              |           ❌              |            ⏳            |


### Approximate Bayesian Methods

| UQ-Method            | Regression            | Classification            | Segmentation              | Pixel Wise Regression      |
|----------------------|:---------------------:|:-------------------------:|:-------------------------:|:--------------------------:|
| BNN_VI_ELBO          |          ✅           |           ✅              |           ✅              |            ⏳            |
| BNN_VI               |          ✅           |           ⏳              |           ⏳              |            ⏳            |
| DKL                  |          ✅           |           ✅              |           ❌              |            ⏳            |
| DUE                  |          ✅           |           ✅              |           ❌              |            ⏳            |
| Laplace              |          ✅           |           ✅              |           ❌              |            ⏳            |
| MC-Dropout           |          ✅           |           ✅              |           ✅              |            ⏳            |
| SGLD                 |          ✅           |           ✅              |           ⏳              |            ⏳            |
| SWAG                 |          ✅           |           ✅              |           ✅              |            ⏳            |
| Deep Ensemble        |          ✅           |           ✅              |           ✅              |            ⏳            |

### Post-Hoc Calibration methods

| UQ-Method            | Regression            | Classification            | Segmentation              | Pixel Wise Regression      |
|----------------------|:---------------------:|:-------------------------:|:-------------------------:|:--------------------------:|
| Conformal QR         |          ✅           |           ❌              |           ❌              |            ⏳              |
| Temperature Scaling  |          ❌           |           ✅              |           ⏳              |            ❌              |

# Tutorials

We try to provide many different tutorials so that users can get a better understanding of implemented methods and get a feel for how they apply to different problems.
Head over to the [tutorials](https://lightning-uq-box.readthedocs.io/en/latest/tutorial_overview.html) page to get started.
# Installation

```console
$ git clone https://github.com/lightning-uq-box/lightning-uq-box.git
$ cd lightning-uq-box
$ pip install .
```

# Documentation 
We aim to provide an extensive documentation on all included UQ-methods that provide some theoretical background, as well as tutorials that illustrate these methods on toy datasets. 