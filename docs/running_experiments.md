(running_experiments)=

# Running Experiments

One of the main motiviations of the Lightning-UQ-Box is to provide an experiment setup where you can quickly launch experiments at scale and control these through config files for reproducability.

## Lightning-CLI

## Alternatives

If you would like to have more control than what the default CLI setup provides, you can also consider a tool like [Hydra](https://hydra.cc/docs/intro/) for experiment configuration. The approach is quiet similar, just
the setup of the config file follows a different notation. For example, given the following config file,

```yaml
uq_method:
  _target_: lightning_uq_box.uq_methods.MCDropoutRegression
  model:
    _target_: lightning_uq_box.models.MLP
    n_outputs: 1
    n_hidden: [50]
    dropout_p: 0.1
  optimizer:
    _target_: torch.optim.Adam
    _partial_: true
    lr: 0.003
  loss_fn:
    _target_: torch.nn.MSELoss
  num_mc_samples: 10
  burnin_epochs: 5

trainer: # lightning trainer arguments
  _target_: lightning.pytorch.Trainer
  accelerator: gpu
  max_epochs: 25

datamodule:
  _target_: lightning_uq_box.datamodules.ToyHeteroscedasticDatamodule
  batch_size: 32
```

You can create the following python code to run experiments:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

conf = OmegaConf.load("path_to_above_config.yaml")
mc_dropout_module = instantiate(conf.uq_method)
datamodule = instantiate(conf.datamodule)
trainer = instantiate(conf.trainer)

trainer.fit(mc_dropout_module, datamodule)
```