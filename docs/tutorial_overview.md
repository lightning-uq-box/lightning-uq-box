(tutorial_overview)=

# Tutorials

The tutorials have two central goals:

1. Briefly introduce the idea of any particular UQ-Method
2. Show how you can apply and adapt these methods for your use case

1D Regression tutorials have the advantage that they are able to illustrate some concepts visually so they are generally a good place to start, however, we aim to provide tutorials for a wide variety of applications. The tutorial notebooks can also be run on google colab, by clicking on the little rocket icon at the top of the page.

## Regression 1D targets

These tutorials present the regression methods on a 1D toy dataset. However, the methods are equally equipped to handle other data modalities, for example image regression by swapping out the underlying network architecture.

```{toctree}
:maxdepth: 1

tutorials/regression
```

## Classification

These tutorials present the classification methods on a Two Moons toy dataset. However, the methods are equally equipped to handle other data modalities, for example image classification by swapping out the underlying network architecture.

```{toctree}
:maxdepth: 1

tutorials/classification
```

## Image Generation

These tutorials cover generative models that learn a distribution over images and can draw novel samples from it. The VAE tutorials use a continuous latent; the VQ-VAE tutorial uses a discrete one and needs a second-stage prior before it can generate at all.

```{toctree}
:maxdepth: 1

tutorials/generation
```

## Earth Observation Applications

```{toctree}
:maxdepth: 1

tutorials/earth_observation
```
