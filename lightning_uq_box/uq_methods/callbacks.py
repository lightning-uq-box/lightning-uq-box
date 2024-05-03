# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Utility Callbacks applicable to multiple methods."""

from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchvision.utils import make_grid

class LogImageSamples(Callback):
    """Callback for logging sampled images during training."""
    def __init__(self, num_samples=8, every_n_steps=5):
        """Initialize the callback.
        
        Args:
            num_samples: The number of samples to generate.
            every_n_steps: The number of steps between logging.
        """
        super().__init__()
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps

    def on_train_step_end(self, trainer: Trainer, pl_module: LightningModule):
        """On training step end, check logging."""

        if self.trainer.global_step % self.log_samples_every_n_steps == 0:
            sampled_imgs = self.generate_imgs(pl_module)
            grid = make_grid(sampled_imgs, nrow=sampled_imgs.shape[0], normalize=True, range=(-1,1))

            import pdb
            pdb.set_trace()
            trainer.logger.experiment.add_image(f"generation_{self.trainer.global_step}", grid, global_step=trainer.current_epoch)
                

    def generate_imgs(self, pl_module: LightningModule) -> Tensor:
        """Generate Image samples from the UQ-Method.

        By default it calls the sampling method of the UQ-Method.

        Can be overwritten by subclasses to generate images in a different way
        and still use the logging functionality.
        
        Args:
            pl_module: The module to generate images from.
        
        """
        return pl_module.sample()

