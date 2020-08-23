import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (VGG, xavier_init, constant_init, kaiming_init,
                      normal_init)

from mmcv.runner import load_checkpoint
from ..registry import BACKBONES


@BACKBONES.register_module
class VGG16(VGG):
    def __init__(self,
                 depth=16,
                 with_last_pool=False,
                 with_bn=False,
                 ceil_mode=True,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(4,),
                 frozen_stages=-1,
                 ):
        super(VGG16, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            with_bn=with_bn,
            dilations=dilations,
            frozen_stages=frozen_stages,
            out_indices=out_indices,
            )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i, num_blocks in enumerate(self.stage_blocks):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                if i == 3 and isinstance(vgg_layer, nn.MaxPool2d):
                    # print(j, vgg_layer)
                    x = x
                else:
                    x = vgg_layer(x)

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

