#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay_backbone = []
    group_decay_decoder = []
    group_no_decay_decoder = []
    group_no_decay_backbone = []
    # for m in module.modules():
    #     if isinstance(m, nn.Linear):
    #         group_decay_backbone.append(m.weight)
    #         if m.bias is not None:
    #             group_no_decay.append(m.bias)
    #     # elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
    #     #     group_decay.append(m.weight)
    #     #     if m.bias is not None:
    #     #         group_no_decay.append(m.bias)
    #     elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
    #         if m.weight is not None:
    #             group_no_decay.append(m.weight)
    #         if m.bias is not None:
    #             group_no_decay.append(m.bias)

    for m in module.named_modules():
        if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
            group_decay_backbone.append(m[1].weight)
        elif 'backbone' in m[0] and (isinstance(m[1], norm_layer) or isinstance(m[1], nn.GroupNorm)):
            if m[1].weight is not None:
                group_no_decay_backbone.append(m[1].weight)
            if m[1].bias is not None:
                group_no_decay_backbone.append(m[1].bias)
        elif 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
            group_decay_decoder.append(m[1].weight)
        elif 'backbone' not in m[0] and (isinstance(m[1], norm_layer) or isinstance(m[1], nn.GroupNorm)):
            if m[1].weight is not None:
                group_no_decay_decoder.append(m[1].weight)
            if m[1].bias is not None:
                group_no_decay_decoder.append(m[1].bias)


    # assert len(list(module.parameters())) == len(group_decay_backbone) + len(group_decay_decoder) + len(group_no_decay)
    weight_group.append(dict(params=group_decay_backbone, lr=lr))
    weight_group.append(dict(params=group_no_decay_backbone, weight_decay=.0, lr=lr))
    weight_group.append(dict(params=group_decay_decoder, lr=10*lr))
    weight_group.append(dict(params=group_no_decay_decoder, weight_decay=.0, lr=10*lr))

    return weight_group
