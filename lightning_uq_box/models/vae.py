# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the MIT License.

"""Adapted from torchseg: https://github.com/isaaccorley/torchseg/blob/main/torchseg/decoders/unet/decoder.py"""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchseg.base.modules import Conv2dReLU, Attention

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x: Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class VAEDecoder(nn.Module):
    def __init__(
        self,
        decoder_channels : int,
        use_batchnorm : bool = True,
        attention_type: bool = None,
    ):
        """Initialize the VAE Decoder.
        
        Args:
            bottleneck_channels: The number of channels in the bottleneck layer.
            decoder_channels: The number of channels in the decoder layers.
            use_batchnorm: Whether to use batch normalization.
            attention_type: The type of attention module to use.
        """
        super().__init__()

        # # computing blocks input and output channels
        dec_in_channels = list(decoder_channels[:-1])
        dec_out_channels = list(decoder_channels[1:])

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, out_ch, **kwargs)
            for in_ch, out_ch in zip(dec_in_channels, dec_out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Decoder.
        
        Args:
            x: The input tensor, which is the encoding of the input image.

        Returns:
            The decoded output tensor.
        """
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x)

        return x