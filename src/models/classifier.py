import logging
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.join(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.modified_UNet import StackedConvLayers, ConvDropoutNormNonlin

log = logging.getLogger(__name__)


class CLFCNN(nn.Module):
    def __init__(self, in_channels: int, channels_conv: List[int], channels_dense: List[int], dropout: float,
                 norm_op=nn.InstanceNorm3d):
        """
        A simple classification net that uses feature maps as inputs. Shallow CNN version.
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels_conv = channels_conv
        self.channels_dense = channels_dense
        self.dropout = dropout
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.norm_op = norm_op

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'kernel_size': 3, 'padding': 1}
        self.conv_blocks = self._create_conv_blocks()
        self.mlp = self._create_mlp()
        self._init_weights()

    def _create_conv_blocks(self):
        blocks = []
        if self.channels_conv:
            blocks.append(StackedConvLayers(self.in_channels, self.channels_conv[0], 2, nn.Conv3d, self.conv_kwargs,
                                            self.norm_op, {'eps': 1e-5, 'affine': True},
                                            nn.Dropout3d, {'p': 0, 'inplace': True},
                                            nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True},
                                            [2, 2, 2], basic_block=ConvDropoutNormNonlin))
            for i in range(len(self.channels_conv) - 1):
                blocks.append(StackedConvLayers(self.channels_conv[i], self.channels_conv[i + 1], 2,
                                                nn.Conv3d, self.conv_kwargs,
                                                self.norm_op, {'eps': 1e-5, 'affine': True},
                                                nn.Dropout3d, {'p': 0, 'inplace': True},
                                                nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True},
                                                [2, 2, 2], basic_block=ConvDropoutNormNonlin))
            return nn.ModuleList(blocks)

    def _create_mlp(self):
        dense_layers = []
        if self.channels_dense:
            if len(self.channels_dense) > 1:
                layer = nn.Sequential(
                    nn.LazyLinear(out_features=self.channels_dense[0], bias=True),
                    nn.BatchNorm1d(self.channels_dense[0]),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                )
                dense_layers.append(layer)
                for i in range(1, len(self.channels_dense) - 1):
                    layer = nn.Sequential(
                        nn.Linear(in_features=self.channels_dense[i - 1], out_features=self.channels_dense[i],
                                  bias=True),
                        nn.BatchNorm1d(self.channels_dense[i]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                    )
                    dense_layers.append(layer)
                last_layer = nn.Sequential(
                    nn.Linear(in_features=self.channels_dense[-2], out_features=self.channels_dense[-1], bias=True),
                )
                dense_layers.append(last_layer)
            else:
                layer = nn.Sequential(
                    nn.LazyLinear(out_features=self.channels_dense[0], bias=True),
                )
                dense_layers.append(layer)
        return nn.ModuleList(dense_layers)

    def _init_weights(self):
        def _init_conv3d_weights(m):
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_uniform_(m.weight)

        for block in self.conv_blocks:
            block.apply(_init_conv3d_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_conv:
            for conv_layer in self.conv_blocks:
                x = conv_layer(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.channels_dense:
            for mlp_layer in self.mlp:
                x = mlp_layer(x)
        return x
