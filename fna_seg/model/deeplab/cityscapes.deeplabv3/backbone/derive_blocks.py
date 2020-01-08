import torch.nn as nn
import torch

from utils.config_utils import parse_net_config
from .operations import OPS, conv_bn


class Block(nn.Module):
    def __init__(self, in_ch, block_ch, ops, stride, dilation=1, use_se=False, bn_params=[True, True]):
        super(Block, self).__init__()
        layers = []

        for i, op in enumerate(ops):
            if i == 0:
                block_stride = stride
                block_in_ch = in_ch
            else:
                block_stride = 1
                block_in_ch = block_ch
            block_out_ch = block_ch
            layers.append(OPS[op](block_in_ch, block_out_ch, block_stride, dilation, 
                                        affine=bn_params[0], track=bn_params[1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def derive_blocks(net_config, width_multi=1.0, output_stride=16):
    parsed_net_config = parse_net_config(net_config)

    current_stride = 1
    rate = 1

    blocks = nn.ModuleList()
    blocks.append(conv_bn(3, parsed_net_config[0][0][0], 2))
    current_stride *= 2

    for cfg in parsed_net_config:
        stride = cfg[2]

        if current_stride == output_stride:
            rate *= stride
            dilation = rate
            stride = 1
        else:
            stride = stride
            dilation = 1
            current_stride *= stride
        blocks.append(Block(int(cfg[0][0]*width_multi), int(cfg[0][1]*width_multi), 
                            cfg[1], stride, dilation))

    return blocks, parsed_net_config
