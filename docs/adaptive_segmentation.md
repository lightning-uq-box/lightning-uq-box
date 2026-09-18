# Adaptive conformal segmentation

AT supports RGB images and binary masks. The fitted base segmentation model emits
raw logits `[B,1,H,W]`; it remains in evaluation mode and receives no gradients.
The threshold network predicts one sigmoid threshold per image. AT regresses the
largest oracle threshold attaining at least `1-alpha` positive-pixel recall.
Discrete recall and tied probabilities can prevent exact attainment.

Use four independent splits: base-model training, threshold-network training,
calibration, and test. Configure the first Trainer explicitly (the constructor's
`max_epochs` is a recommendation, not a Trainer setting):

```python
from lightning import Trainer
from lightning_uq_box.uq_methods import AdaptiveThresholding

method = AdaptiveThresholding(model=fitted_binary_unet, alpha=0.1)
Trainer(max_epochs=30).fit(method, train_dataloaders=threshold_train_loader)
Trainer(max_epochs=1).fit(method, train_dataloaders=calibration_loader)
Trainer().test(method, dataloaders=test_loader)
```

Calibration uses image-weighted recall with target `(n+1)/n * (1-alpha)`;
the correction is **subtracted**: `tau = clip(predicted_tau - t_prime, 0, 1)`.
If the corrected target exceeds one, the safe fallback predicts all pixels as
foreground. Empty foreground masks have hard recall one. The reported coverage
gap is the **mean of per-image absolute gaps**, not the absolute gap of the mean.
Calibration requires one process. Under exchangeability the method controls
marginal false-negative risk; it does not provide unconditional image-conditional
coverage guarantees. Multiclass and false-positive control are outside this scope.

Prediction returns `pred` and `phat` maps `[B,1,H,W]`, and `tau` and `pred_uct`
thresholds `[B]`. `pred_uct` is the threshold, not an uncertainty probability.
With `save_preds=True`, maps and targets are HDF5 datasets and scalar thresholds
are HDF5 attributes. Checkpoints preserve both stages and the calibration shift.
No medical-dataset reproduction result is claimed here.

The implementation follows the paper specification recorded in the implementation
plan: a single linear ResNet head, subtraction correction, and finite-sample
calibration. The current [released scripts](https://github.com/bjbbbb/Conditional-Optimization-for-Adaptive-Thresholding)
use a two-layer head and a differently signed correction without this finite-sample
target. Their reported metrics assign empty masks recall zero while their training
soft recall assigns one; here hard recall consistently assigns one. These choices
must be disclosed when comparing empirical results. See the
[paper](https://openreview.net/forum?id=Gd2AiWes1J).

## COAT

Replace AT with `COAT(model=fitted_binary_unet, alpha=0.1, temperature=0.05)`
and train the threshold network for 60 epochs with learning rate `5e-4` (defaults).
Calibration and test calls remain identical. COAT learns directly from masks:
`sigmoid((phat - tau) / temperature)` relaxes the hard foreground set, and the
loss is the mean squared gap between each image's soft recall and `1-alpha`.
The denominator adds `eps=1e-6`; empty masks contribute a constant loss with
zero gradient. This soft-loss convention differs from the hard recall convention
(one for empty masks). The vectorized formula follows the paper's epsilon loss;
the live reference code uses an explicit empty-mask branch instead.
