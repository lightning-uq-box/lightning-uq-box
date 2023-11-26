<img src="docs/_static/logo.png" alt="Lightning-UQ-Box logo" width="200" height="150" />

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

## Installation

```console
$ git clone https://github.com/lightning-uq-box/lightning-uq-box.git
$ cd lightning-uq-box
$ pip install .
```

## Documentation 
We aim to provide an extensive documentation on all included UQ-methods that provide some theoretical background, as well as tutorials that illustrate these methods on toy datasets. 