# uq-method-box

A set of datasets and models to evaluate predictive uncertainty on regression tasks. As a PyTorch library it relies heavily on PytorchLightning and GPytorch.
The goal of this library is threefold:

1. Provde access to a variety of Uncertainty Quantification methods for Modern Deep Neural Networks that work with a range of neural network architectures.
2. Make it easy to compare methods on a new dataset or a new method on existing datasets.
3. Focus on reproducibility of experiments with miminum boiler plate code and standardized evaluation protocols.

## Motivation
The motivation for this library was to assemble a variety of Uncertainty Quantification Methods for Deep Learning in a common framework. This library implements a subset of the methods included in [A Survey of Uncertainty in Deep Neural Networks (Gawlikowski et al. 2022)](https://arxiv.org/abs/2107.03342) as well as some additional methods that have come out since then or were not explicitly covered. To this point we have solely focused on regression tasks, but as an open-source library we of course welcome contributions for additional methods or extensions. Each implemented method is covered in a separate notebook that explains some of the underlying theory as well as a toy example to demonstrate how you might be able to use this library in your own research projects.

## Design Principles

The two driving design principles behind the library twofold:
1. To provide a variety of UQ-methods in a common framework such that different methods can be tried out and accelerate the turnover of experiment pipelines.
2. Focus on reproducibilty so that experiments can be easily verified without having to rewrite a lot of code

To this end, we heavily rely on [Lightning](https://lightning.ai/docs/pytorch/stable/) for the implementation of UQ-methods as [LightningModules](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) and datasets as [LightningDataModules](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). Training and evaluation can then easily be conducted via the [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html). We think that this still gives enough flexibility for small projections or just trying out different methods quickly, while also providing a reliable framework for larger research experiments. We encourage the use of [Hydra](https://hydra.cc/) as a way to configure experiments and in that manner clearly define the used hyperparameters so that they can be easily reproduced by different sources.

## Documentation 
We aim to provide an extensive documentation on all included UQ-methods that provide some theoretical background, as well as tutorials that illustrate these methods on toy datasets. 

## Experiments
The experiments conducted for our paper "Title here." are contained in the [experiments](experiments/) directory. To reproduce the results in our paper follow these instructions.

## Contribution
For a basic contribution guide see this [doc](contribution_guide.md).

## Citation
If you use this software in your work, please cite...