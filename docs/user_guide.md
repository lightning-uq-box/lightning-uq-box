(guide)=

# Why lightning-uq-box

Over the past decade open-sourcing code alongside publications has luckily become a de facto standard. However, different papers continue to reimplement and develop methods in their own frameworks with a lot of boilerplate code. This can make it difficult to understand what the core aspects of a proposed method are. Additionally, this makes it more time-consuming to test various methods on your task of interests as each implementation might have different requirements and custom functionality. We aim to bridge this gap with the lightning-uq-box, which is essentially nothing more than a collection of [Lightning Modules](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) that build on the shoulders of existing implementations and packages and give a common structure across methods.

With that comes the integration of [Lightning-CLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) which lets you run experiments at scale from the command line through configuration files to ensure reproducibility.

## Library Design Choice

The following aims to explain the design choices behind the implementations found in the Lightning-UQ-Box. We will first give some "definitions" about how we are using the three terms "UQ-Method", "Application Task", "Task Data Modalities".


### UQ-Method
- a methodology developed that can be applied to a Neural Network and extract or
  enhance predictive uncertainty estimates
- the methodology can be of different nature, for example:
  - modeling network weights as distributions like BNNs
  - post-hoc method based on pretrained deterministic networks like Laplace or
    Conformal Prediction
  - sampling methods like Diffusion models
  - specific network architectures and loss functions like Deep Evidential
    Regression
- however, there are also commonalities across methods
  - how regression output from samples are computed is the same for MCDropout,
    SWAG, Ensembles etc (Gaussian moment matching)

### Application Task
- a task that is defined by the problem setting for which you have labels
- regression,classification, segmentation, pixel wise regression, object
  detection, change detection etc.
- an application task might have different demands given a certain method
- however, there are also several commonalities across a subset of methods
  given any particular application task

### Task Data Modalities
- a specific application task can involve different data modalities
- regression task could be vector inputs and single target output, or image
  input with single target output, or image input with multiple regression
  targets etc.

## Design

- modular design aiming to:
  - give enough structure to apply any particular method to a
    dataset that practitioner might be interested in
  - give enough flexibility to extend or adapt for particular problem with minimal
    changes
  - support any model architecture for any particular UQ-Method (like pretrained timm models)
- each UQ-method has its own "Base Class" that implements the commonalities across tasks, for example MCDropoutBase
- for each different task, the UQ-method will have a subclass specific for that
  task so MCDropoutClassification, MCDropoutRegression etc
- therefore, we incur code duplication but we prefer this for clarity of separation and avoiding long if/else blocks
  for any particular method
- the common features across tasks like the metrics for example, or
  functionality like computing the mean, can be methods that are forced to be
  implemented in the subclasses but can be util functions that could still be utilized by users for their own usage
- try to integrate existing implementations into this framework to not reinvent the wheel

## Pytorch and Lightning
- use [PyTorch](https://pytorch.org/docs/stable/index.html), because:
  - wide adaptation in research community
  - ready to use architectures and pretrained models for example through [timm library](https://github.com/huggingface/pytorch-image-models)
    or [torchseg](https://github.com/isaaccorley/torchseg)
- use [Lightning](https://lightning.ai/docs/pytorch/stable/), because:
  - reduces boiler plate code
  - enforces code organization structure
  - offers flexibility to introduce functionality, like early stopping, precision
  - reproducibility through standardization
  - CLI integration for running code from config files for reproducibility and scale
