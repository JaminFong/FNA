# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from backbone.derived_backbone import BackBone
from head.ASPP_Sep import ASPP_Sep
from seg_opr.seg_oprs import AttentionRefinement, ConvBnRelu, FeatureFusion


class DeepLabV3(nn.Module):
    def __init__(self, is_training, config, criterion=None,
                 norm_layer=nn.BatchNorm2d):
        super(DeepLabV3, self).__init__()
        input_channel = 320

        self.business_layer = []

        self.backbone = BackBone(config, is_training=is_training)
        self.head = ASPP_Sep(dim_in=input_channel, dim_out=config.MODEL_ASPP_OUTDIM, 
                            BatchNorm=norm_layer, rate=16//config.MODEL_OUTPUT_STRIDE)
        self.cls_conv = nn.Conv2d(in_channels=config.MODEL_ASPP_OUTDIM, out_channels=config.num_classes, 
                            kernel_size=1, stride=1, padding=0)

        self.business_layer.append(self.head)
        self.business_layer.append(self.cls_conv)

        self.is_training = is_training

        if is_training:
            self.criterion = criterion

    def forward(self, data, label=None):
        result = self.backbone(data)
        result = self.head(result)
        result = self.cls_conv(result)
        result = nn.functional.interpolate(result, size=data.size()[2:], mode='bilinear', align_corners=True)
        if self.is_training:
            loss = self.criterion(result, label)
            return loss
        return F.log_softmax(result, dim=1)
