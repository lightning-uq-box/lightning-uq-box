# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
# ---

# # Evaluation of Predictive Uncertainty

# ## Accuracy Metrics
#
# Below a range of accuracy metrics are plotted, to evaluate the different networks on the test set.
#
# * mean absolute error, $mae = \frac{1}{n} \sum_{i=1}^n|f_{\theta}(x_i)-y_i|$.
# * root mean squared error, $rmse = \frac{1}{n} \sum_{i=1}^n|f_{\theta}(x_i)-y_i|^2$.
# * median absolute error, $mdae = \text{median}((|f_{\theta}(x_i)-y_i|)_{i=1}^n$
# * coefficient of determination, $R^2 = 1 - (\sum_{i=1}^n (y_i - f_{\theta}(x_i))^2)/\sigma_y$ where $\sigma_y = \sum_{i=1}^n(y_i - \bar{y})^2$ and $\bar{y} = \frac{1}{n}\sum_{i=1}^ny_i$
# * correlation, $corr = Corr((f(x_i))_{i=1}^n,(y_i)_{i=1}^n)$.
#
# Remarks:
#
# * For mae, rmse, mdae optimal values are near zero.
# * For $R^2$ and corr optimal value are near $1$.

# ## Scoring Rules
#
# A proper scoring rule is a real-valued
# function that takes a predictive distribitution $p_{\theta}(y|x)$ and an event $y_i$ from the true distribution
# $q(y|x)$ as inputs and produces a numerical value that is only minimized if the distributions are exactly
# equal. In other words, a proper scoring rule attains its optimal score if the predictive distribution matches the ground truth distribution exactly.  In the following we consider negatively oriented scores, meaning that the if the score obtains the minimum the predictive distribution matches the ground truth distribution. In other words, a smaller score means that the predictive distribution matches the true distribution better. For an in depth treatmeant of the used scoring rules see [Strictly Proper Scoring Rules, Prediction,
# and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
#
# #### Negative Log Likelihood (NLL)
#
# Negative log likelihood for a gaussian probability distribution $p_{\theta}(y|x)$ which is also used as a loss functions in some methods. Note that the probability distribution have values between $0$ and $1$. If given a certain $x$, $p_{\theta}(y|x) \approx 1$, we could almost be certain that our network predicts the label $y$. Thus, we minimize the negative logarithm of the likelihood, as $log(1) = 0$ and $log(a) \to \infty$ for $a \to 0$.
#
# Note that the negative log likelihood also has some disadvantages:
# "Commonly used to evaluate the quality of model uncertainty on some held out set. Drawbacks: Although a proper scoring rule ,[Strictly Proper Scoring Rules, Prediction,
# and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), it can over-emphasize tail probabilities, [Evaluating Predictive Uncertainty Challenge](https://quinonero.net/Publications/quinonero06epuc.pdf).", [Evaluating
# Predictive Uncertainty Under Dataset Shift](https://proceedings.neurips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)
#
# #### Continuous Ranked Probability Score (CRPS)
# The negatively oriented continuous ranked probability score for Gaussians. Where negatively oriented means a smaller value is more desirable. The negatively oriented crps for a Gaussian $\mathcal{N}(\mu, \sigma)$, which is the distribution used for prediction, and $y$ is the label, is given by,
#
# $crps(\mathcal{N}(\mu, \sigma), y) = -\sigma \big(\frac{y-\mu}{\sigma}(2\Phi(\frac{y-\mu}{\sigma})-1)+2\phi(\frac{y-\mu}{\sigma})-\frac{1}{\sqrt{\pi}}\big),$
#
# where $\Phi$ is the cumulative density function and $\phi$ probability distribution of a Gaussian with mean $0$ and variance $1$. For more details see [Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and
# Minimum CRPS Estimation](https://sites.stat.washington.edu/MURI/PDF/gneiting2005.pdf)
#
# Then, we compute the average sum over all predictions and labels, where $f_{\theta}(x_i) = (\mu(x_i), \sigma(x_i))$,
#
# $CRPS = \frac{1}{n} \sum_{i=1}^n crps(f_{\theta}(x_i), y_i).$
#
# For further reading, see for example, [Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems](https://journals.ametsoc.org/view/journals/wefo/15/5/1520-0434_2000_015_0559_dotcrp_2_0_co_2.xml )
#
# The following scoring rules, are scoring rules for quantiles. Note that given a Gaussian distribution, one can compute the corresponding quantiles. For an in depth treatmeant see [Strictly Proper Scoring Rules, Prediction,
# and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
#
# #### Check Score (Check)
# The negatively oriented check score, also known as pinball loss.
# In short the check score computes the sum of the differences between a chosen set of quantiles of the predictive distribution and the true labels.
#
# #### Interval Score (Interval)
# The negatively oriented interval score.

# ## Calibration
#
# Calibration refers to the degree to which a predicted distribution matches the true underlying distribution of the data. Recall that for regression, our neural network either outputs quantiles given an input $x$ or a mean and variance of a probability distribution, which can be converted to quantiles.
#
# Formally, [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf), one can say that our model's predictions are well calibrated if
#
# $\frac{1}{n}\sum_{i=1}^n \mathbb{I}(y_i \leq F^{-1}_i(p)) \to p$ for all $p \in [0,1]$, as the number of data points $n \to \infty$.
#
# Here $\mathbb{I}(y_i \leq F^{-1}_i(p)) = 1$ if $y_i \leq F^{-1}_i(p)$ and zero otherwise. Moreover, $F_i(y) := P(Y \leq y \vert x_i) \in [0,1]$ and $F^{-1}_i(p) = \inf \{ y \in \mathbb{R}: p \leq F_i(y) \}$ denotes the quantile function.
#
# In simpler words, the empirical and the predicted
# cumulative distribution functions should match, as the dataset size goes to infinity.
#
#  Below we plot a range of calibration metrics.
#
#
#
# *   Root-mean-squared Calibration Error, rms_cal, gives the root mean squared error of the expected and observed proportions for a given range of quantiles.
# *   Mean-absolute Calibration Error,  ma_cal, gives the mean absolute error of the expected and observed proportions for a given range of quantiles.
# *   Miscalibration Area, miscal_area, if the number of quantiles is the same this is just mean absolute calibration error. Otherwise as the number of quantiles increases this describes area between the curves of predicted and empirical quantiles.
#
# Remarks:
#
# For all calibration metrics a lower value indicates a better calibration of our model.

# ## Sharpness
#
# Sharpness captures how concentrated the predictive distribution is around its mean and is a property of the predictive distribution only, meaning that it doesn not depend on the ground truth observations.
#
# Below a sharpness metric is computed, it returns single scalar which quantifies the average of the standard deviations predicted by our model, $\sigma_{\theta}$.
#
# It is given by,
#
# sharp $= \sqrt{\frac{1}{n} \sum_{i=1}^n \sigma_{\theta}(x_i)^2}.$
#
# Note, that this metric does not compare the empirical variance of the labels, but only yields a measure on our predictive uncertainties.

#
