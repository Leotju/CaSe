import torch.nn as nn
from ..registry import HEADS


@HEADS.register_module
class CountHead(nn.Module):
    def __init__(self,
                 roi_feat_size=7,
                 in_channels=512,
                 fc_out_channels=1024,
                 ):
        super(CountHead, self).__init__()
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels

        in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fc_count = nn.Sequential(
            nn.Linear(in_channels, fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out_channels, fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(fc_out_channels, 1),
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        count_pred = self.fc_count(x)
        return count_pred



