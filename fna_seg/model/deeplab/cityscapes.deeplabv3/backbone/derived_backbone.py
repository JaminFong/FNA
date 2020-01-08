import math
import random
import re
import sys

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.param_remap import remap_for_paramadapt
from .derive_blocks import derive_blocks


class BackBone(nn.Module):
    def __init__(self, config,  is_training=True):
        super(BackBone, self).__init__()
        self.blocks, _ = derive_blocks(config.net_config, config.width_multi, config.MODEL_OUTPUT_STRIDE)
        self.config = config
        self._initialize_weights()
        if is_training:
            remap_for_paramadapt(config.load_path, self.state_dict(), config.seed_num_layers)

    def forward(self, x):

        for i, block in enumerate(self.blocks):
            x = block(x)
        return x


    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                assert self.config.backbone_init in ['normal', 'kaiming']
                if self.config.backbone_init == 'normal':
                    torch.nn.init.normal_(m.weight,0,0.01)
                elif self.config.backbone_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.eps = self.config.backbone_bn_eps
                m.momentum = self.config.backbone_bn_momentum
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
