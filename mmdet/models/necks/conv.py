import torch.nn as nn
from mmcv.cnn import xavier_init
from ..registry import NECKS
from ..utils import ConvModule



@NECKS.register_module
class Conv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            activation=activation
        )

    def forward(self, inputs):
        outs = self.conv(inputs[0])
        return tuple([outs])
