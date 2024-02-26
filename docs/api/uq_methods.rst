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

Deep Evidential Regression
--------------------------

.. currentmodule:: lightning_uq_box.uq_methods.deep_evidential_regression

.. autoclass:: DER

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


Stochastic Gradient Langevin Dynamics (SGLD)
--------------------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.sgld

.. autoclass:: SGLD

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


Hierachical Probabilistic UNet
------------------------------

.. currentmodule:: lightning_uq_box.uq_methods.hierachical_prob_unet

.. autoclass:: HierachicalProbUNet

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