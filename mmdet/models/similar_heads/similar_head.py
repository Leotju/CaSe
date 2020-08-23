import torch.nn as nn
from ..registry import HEADS


@HEADS.register_module
class SimilarHead(nn.Module):
    def __init__(self,
                 roi_feat_size=7,
                 in_channels=512,
                 num_fc=2,
                 fc_out_channels=1024,
                 embed_dims=64,
                 ):
        super(SimilarHead, self).__init__()
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.embed_dims = embed_dims
        in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fc_se = nn.ModuleList()
        for i in range(num_fc):
            if i == 0:
                self.fc_se.append(nn.Linear(in_channels, fc_out_channels))
            else:
                self.fc_se.append(nn.Linear(fc_out_channels, fc_out_channels))
            self.fc_se.append(nn.ReLU(inplace=True))
        self.fc_se.append(nn.Linear(fc_out_channels, embed_dims))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc_se in self.fc_se:
            x = fc_se(x)
        return x

