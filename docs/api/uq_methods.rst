lightning_uq_box.uq_methods
===========================

Single Forward Pass Methods
===========================

Mean Variance Estimation
------------------------

.. currentmodule:: lightning_uq_box.uq_methods.mean_variance_estimation

Mean Variance Base
``````````````````

.. autoclass:: MVEBase

Mean Variance Regression
````````````````````````

.. autoclass:: MVERegression

Quantile Regression
-------------------

.. currentmodule:: lightning_uq_box.uq_methods.quantile_regression

Quantile Regression Base
````````````````````````

.. autoclass:: QuantileRegressionBase

Quantile Regression
```````````````````

.. autoclass:: QuantileRegression

Quantile Pixelwise Regression
`````````````````````````````

.. autoclass:: QuantilePxRegression

Deep Evidential Regression
--------------------------

.. currentmodule:: lightning_uq_box.uq_methods.deep_evidential_regression

Deep Evidential Regression
``````````````````````````
.. autoclass:: DER

Deep Evidential Pixelwise Regression
````````````````````````````````````

.. autoclass:: DERPxRegression

Zig Zag
-------

.. currentmodule:: lightning_uq_box.uq_methods.zigzag

Zig Zag Base
````````````

.. autoclass:: ZigZagBase

Zig Zag Regression
``````````````````

.. autoclass:: ZigZagRegression

Zig Zag Classification
``````````````````````

.. autoclass:: ZigZagClassification


Mixture Density Networks
------------------------

.. currentmodule:: lightning_uq_box.uq_methods.mixture_density

Mixture Density Regression
``````````````````````````

.. autoclass:: MDNRegression


Approximate Bayesian Methods
============================

Monte Carlo Dropout
-------------------

.. currentmodule:: lightning_uq_box.uq_methods.mc_dropout

MC-Dropout Base
```````````````

.. autoclass:: MCDropoutBase

MC-Dropout Regression
`````````````````````

.. autoclass:: MCDropoutRegression

MC-Dropout Classification
`````````````````````````

.. autoclass:: MCDropoutClassification

MC-Dropout Segmentation
```````````````````````

.. autoclass:: MCDropoutSegmentation

MC-Dropout Pixelwise Regression
```````````````````````````````

.. autoclass:: MCDropoutPxRegression

Laplace Approximation
---------------------

.. currentmodule:: lightning_uq_box.uq_methods.laplace_model

Laplace Base
````````````

.. autoclass:: LaplaceBase

Laplace Regression
``````````````````
.. autoclass:: LaplaceRegression

Laplace Classification
``````````````````````

.. autoclass:: LaplaceClassification

Bayesian Neural Networks ELBO
-----------------------------

.. currentmodule:: lightning_uq_box.uq_methods.bnn_vi_elbo

BNN ELBO Base
`````````````

.. autoclass:: BNN_VI_ELBO_Base

BNN ELBO Regression
```````````````````

.. autoclass:: BNN_VI_ELBO_Regression

BNN ELBO Classification
```````````````````````
.. autoclass:: BNN_VI_ELBO_Classification

BNN ELBO Segmentation
`````````````````````
.. autoclass:: BNN_VI_ELBO_Segmentation

Bayesian Neural Networks with Alpha Divergence
----------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.bnn_vi

BNN VI Base
```````````

.. autoclass:: BNN_VI_Base

BNN VI Regression
`````````````````

.. autoclass:: BNN_VI_Regression

BNN VI Batched Regression
`````````````````````````

.. autoclass:: BNN_VI_BatchedRegression


Bayesian Neural Networks with Latent Variables (BNN-LV)
-------------------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.bnn_lv_vi

BNN LV VI Base
``````````````

.. autoclass:: BNN_LV_VI_Base
.. autoclass:: BNN_LV_VI_Batched_Base

BNN LV VI Regression
````````````````````

.. autoclass:: BNN_LV_VI_Regression
.. autoclass:: BNN_LV_VI_Batched_Regression


Stochastic Weight Averaging Gaussian (SWAG)
-------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.swag

SWAG Base
`````````

.. autoclass:: SWAGBase

SWAG Regression
```````````````

.. autoclass:: SWAGRegression

SWAG Classification
```````````````````

.. autoclass:: SWAGClassification

SWAG Segmentation
`````````````````

.. autoclass:: SWAGSegmentation

SWAG Pixelwise Regression
`````````````````````````

.. autoclass:: SWAGPxRegression


Stochastic Gradient Langevin Dynamics (SGLD)
--------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.sgld

.. autoclass:: SGLD

Variational Bayes Last Layer
----------------------------

.. currentmodule:: lightning_uq_box.uq_methods.vbll

VBLL Classification
```````````````````

.. autoclass:: VBLLClassification

VBLL Regression
```````````````

.. autoclass:: VBLLRegression

Spectral Normalized Gaussian Process (SNGP)
-------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.sngp

SNGPBase
````````
.. autoclass:: SNGPBase

SNGPRegression
```````````````

.. autoclass:: SNGPRegression

SNGPClassification
``````````````````
.. autoclass:: SNGPClassification

Deep Kernel Learning (DKL)
--------------------------

.. currentmodule:: lightning_uq_box.uq_methods.deep_kernel_learning

DKL Base
````````

.. autoclass:: DKLBase

DKL Regression
``````````````

.. autoclass:: DKLRegression

DKL Classification
``````````````````

.. autoclass:: DKLClassification


Deterministic Uncertainty Estimation (DUE)
------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.deterministic_uncertainty_estimation

DUE Regression
``````````````

.. autoclass:: DUERegression

DUE Classification
``````````````````

.. autoclass:: DUEClassification


Deep Ensembles
--------------

.. currentmodule:: lightning_uq_box.uq_methods.deep_ensemble

Deep Ensemble Base
``````````````````

.. autoclass:: DeepEnsemble

Deep Ensemble Regression
````````````````````````

.. autoclass:: DeepEnsembleRegression

Deep Ensemble Classification
````````````````````````````

.. autoclass:: DeepEnsembleClassification

Deep Ensemble Segmentation
``````````````````````````

.. autoclass:: DeepEnsembleSegmentation

Deep Ensemble Pixelwise Regression
``````````````````````````````````

.. autoclass:: DeepEnsemblePxRegression


Masked Ensemble
---------------

.. currentmodule:: lightning_uq_box.uq_methods.masked_ensemble

Masked Ensemble Base
````````````````````

.. autoclass:: MasksemblesBase

Masked Ensemble Regression
``````````````````````````

.. autoclass:: MasksemblesRegression

Masked Ensemble Classification
``````````````````````````````

.. autoclass:: MasksemblesClassification


Epinet
------

.. currentmodule:: lightning_uq_box.uq_methods.epinet

Epinet Base
```````````

.. autoclass:: EpinetBase

Epinet Regression
`````````````````

.. autoclass:: lightning_uq_box.uq_methods.loss_functions.EpinetGaussianNoiseLoss

.. autoclass:: EpinetRegression

Epinet Classification
`````````````````````

.. autoclass:: EpinetClassification


Density Uncertainty Model
-------------------------

.. currentmodule:: lightning_uq_box.uq_methods.density_uncertainty

Density Uncertainty Base
````````````````````````

.. autoclass:: DensityLayerModelBase

Density Uncertainty Regression
``````````````````````````````

.. autoclass:: DensityLayerModelRegression

Density Uncertainty Classification
``````````````````````````````````

.. autoclass:: DensityLayerModelClassification


Generative Models
=================

Classification and Regression Diffusion (CARD)
----------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.cards

CARD Base
`````````

.. autoclass:: CARDBase

CARD Regression
```````````````

.. autoclass:: CARDRegression

CARD Classification
```````````````````

.. autoclass:: CARDClassification

Noise Scheduler
```````````````

.. autoclass:: NoiseScheduler


Probabilistic UNet
------------------

.. currentmodule:: lightning_uq_box.uq_methods.prob_unet

.. autoclass:: ProbUNet



Hierarchical Probabilistic UNet
-------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.hierarchical_prob_unet

.. autoclass:: HierarchicalProbUNet

.. autofunction:: hierarchical_ce_loss

This SMP adaptation of `Kohl et al. (2019) <https://arxiv.org/abs/1905.13077>`_
uses separate image prior and image/segmentation posterior encoders. Posterior
samples condition the prior at every latent scale for reconstruction and KL.
Prediction samples only the prior. ``mean`` can be a boolean per scale to explore
coarse-to-fine interventions.

GECO is the default: reconstruction CE is summed over selected pixels and
averaged over images; ``kappa`` is scaled by the selected pixel count. Hard pixel
mining selects 2% over the whole batch with Gumbel-perturbed ranking. The moving
average uses a straight-through reconstruction gradient. The multiplier is a
checkpointed buffer updated once per training batch, independently of optimizer
and scheduler, and synchronized across distributed ranks. A sustained multiplier
at its 1e5 cap suggests an unattainable reconstruction target. Set
``loss_type="elbo"`` for fixed-beta training. Binary tasks use one sigmoid logit;
multiclass tasks use one logit per class.

For a small dataset, use ``encoder_weights="imagenet"`` with
``freeze_backbone=True`` to train the latent hierarchy with frozen encoders.
``freeze_decoder=True`` freezes only the prior's deterministic stitching tail
and segmentation head; stochastic stages remain trainable.

The SMP backbone and decoder differ from the paper's eight-scale residual
architecture. ``blocks_per_level`` and ``convs_per_block`` control residual
refinement before each latent head; SMP blocks perform upsampling and skip
fusion. The posterior ends at its last distribution, and training omits the
unused free-prior pass from the reference. Report parameter counts and training
settings when comparing to the paper; paper-level LIDC scores are not implied
by architecture or smoke tests.

Distributional evaluation uses integer maps ``[B, N, H, W]``, obtained by argmax
of individual sampled logits (threshold for binary). The functions in
``lightning_uq_box.eval_utils`` provide reconstruction IoU, squared GED and
balanced Hungarian IoU with empty-vs-empty IoU equal to one. Always report the
GED diversity term ``d_ss`` alongside GED. For non-divisible sample/grader counts,
Hungarian evaluation repeats both sets to their least common multiple.

UQ Calibration Methods
======================

Test Time Augmentation (TTA)
----------------------------

.. currentmodule:: lightning_uq_box.uq_methods.inference_time_augmentation

.. autoclass:: TTABase
.. autoclass:: TTAClassification
.. autoclass:: TTARegression

Conformal Quantile Regression
-----------------------------

.. currentmodule:: lightning_uq_box.uq_methods.conformal_qr

.. autoclass:: ConformalQR

Temperature TempScaling
-----------------------

.. currentmodule:: lightning_uq_box.uq_methods.temp_scaling

.. autoclass:: TempScaling

Regularized Adaptive Prediction Sets (RAPS)
-------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.raps

.. autoclass:: RAPS

Image to Image Conformal
------------------------

.. currentmodule:: lightning_uq_box.uq_methods.img2img_conformal

.. autoclass:: Img2ImgConformal
