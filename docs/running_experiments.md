(running_experiments)=

# Running Experiments

One of the main motiviations of the Lightning-UQ-Box is to provide an experiment setup where you can quickly launch experiments at scale and control these through config files for reproducability.

## Lightning-CLI

The [Lightning-CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) allows running experiments

```yaml
model: # 'model' is keyword for Lightning-CLI model used in Trainer
  class_path: lightning_uq_box.uq_methods.MCDropoutRegression
  init_args:
    model:
      class_path: lightning_uq_box.models.MLP
      init_args:
        n_outputs: 1
        n_hidden: [50]
        dropout_p: 0.1
        activation_fn:
          class_path: torch.nn.ReLU
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: 0.003
    loss_fn:
      class_path: torch.nn.MSELoss
    num_mc_samples: 10
    burnin_epochs: 1

data: # datamodule arguments
  class_path: lightning_uq_box.datamodules.ToyHeteroscedasticDatamodule
  init_args:
    batch_size: 64

trainer: # pytorch lightning trainer arguments
  max_epochs: 2
```

Then you launch this run via the command line with:

```bash
uq-box fit --config path_to_above_config.yaml
```

You can of course also create your own Lightning-CLI setup (check their [docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for additional features).

If you would like to use the instantitated Lightning classes in your code, but still use config files you can do:

```python
from lightning_uq_box.main import get_uq_box_cli

cli = get_uq_box_cli(["--config", "path_to_above_config.yaml"])
cli.trainer.fit(cli.model, cli.datamodule)
cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
```

## Alternatives

You can also consider a tool like [Hydra](https://hydra.cc/docs/intro/) for experiment configuration. The approach is quiet similar, just
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