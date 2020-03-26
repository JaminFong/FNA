import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import constant_init, normal_init
from mmdet.models.anchor_heads.ssd_head import SSDHead
from mmdet.models.registry import HEADS

from .operations import conv_dw_head


@HEADS.register_module
class SSDLightHead(SSDHead):

    def __init__(self,
                input_size=300,
                num_classes=81,
                in_channels=(512, 1024, 512, 256, 256, 256),
                anchor_strides=(8, 16, 32, 64, 100, 300),
                basesize_ratio_range=(0.1, 0.9),
                anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                target_means=(.0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0),
                search=False):
        super(SSDLightHead, self).__init__(input_size, num_classes, in_channels, 
                                anchor_strides, basesize_ratio_range, anchor_ratios,
                                target_means, target_stds)
        self.search = search
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                conv_dw_head(in_channels[i], num_anchors[i] * 4)
            )
            cls_convs.append(
                conv_dw_head(in_channels[i], num_anchors[i] * num_classes)
            )
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.03)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, feats):
        if self.search:
            feats, net_sub_obj = feats
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        if self.search:
            return (cls_scores, bbox_preds), net_sub_obj 
        else:
            return cls_scores, bbox_preds