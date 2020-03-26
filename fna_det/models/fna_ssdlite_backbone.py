import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from mmdet.models.registry import BACKBONES

from .fna_base_backbone import BaseBackbone
from .operations import OPS, Conv1_1, conv_dw


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, dilation, primitives, 
                    affine=False, track=False, connect_head=False):
        super(MixedOp, self).__init__()
        self.connect_head = connect_head
        self._ops = nn.ModuleList()
        self.primitives = primitives
        for primitive in primitives:
            op = OPS[primitive](C_in, C_out, stride, dilation, affine, track)
            self._ops.add_module('{}'.format(primitive), op)

    def forward(self, x, weights, branch_indices, mixed_sub_obj):
        op_weights = torch.stack([weights[branch_index] for branch_index in branch_indices])
        op_weights = F.softmax(op_weights/0.9, dim=-1)
        if self.connect_head:
            head_data = []
            for branch_index in branch_indices:
                head_data.append(getattr(self._ops, self.primitives[branch_index]).op[:3](x))

            # backbone_data, sub_obj, head_data
            return sum(op_weight * getattr(self._ops, self.primitives[branch_index]).op[3:](
                    head_data[i]) for i, (branch_index, op_weight) in enumerate(zip(branch_indices, op_weights))), \
                        sum(op_weight * mixed_sub_obj[branch_index] for branch_index, op_weight in zip(
                            branch_indices, op_weights)), \
                        sum(op_weight * head_d for op_weight, head_d in zip(op_weights, head_data))
        else:
            return sum(op_weight * getattr(self._ops, self.primitives[branch_index])(x) for branch_index, op_weight in zip(
                        branch_indices, op_weights)), \
                    sum(op_weight * mixed_sub_obj[branch_index] for branch_index, op_weight in zip(
                        branch_indices, op_weights))


class Block(nn.Module):
    def __init__(self, C_in, C_out, stride, num_layer, prim_reduce, prim_norm, bn_params=[True, True], connect_head=False):
        super(Block, self).__init__()
        self.connect_head = connect_head
        self.layers = nn.ModuleList() 
        for inner_idx in range(num_layer):
            if inner_idx == 0:
                self.layers.append(MixedOp(C_in, C_out, stride, 1, 
                                            prim_reduce, 
                                            affine=bn_params[0], track=bn_params[1], 
                                            connect_head=connect_head))
            else:
                self.layers.append(MixedOp(C_out, C_out, 1, 1, 
                                            prim_norm, 
                                            affine=bn_params[0], track=bn_params[1]))

    def forward(self, x, weights_normal, weights_reduce, branch_index, block_sub_obj):
        weights = []
        for weight in weights_reduce:
            weights.append(weight)
        for weight in weights_normal:
            weights.append(weight)

        count_sub_obj = []
        for i, (layer, weight, branch_idx, layer_sub_obj) in enumerate(zip(
                    self.layers, weights, branch_index, block_sub_obj)):
            if i==0 and self.connect_head:
                x, sub_obj, head_data = layer(x, weight, branch_idx, layer_sub_obj)
            else:
                x, sub_obj = layer(x, weight, branch_idx, layer_sub_obj)
            count_sub_obj.append(sub_obj)
        if self.connect_head:
            return x, sum(count_sub_obj), head_data
        else:
            return x, sum(count_sub_obj)


@BACKBONES.register_module
class SSDLiteBackbone(BaseBackbone):
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 64, 'S', 128),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self, input_size, search_params, output_indices=(6, 8)):
        super(BaseBackbone, self).__init__()
        self.output_indices = output_indices
        self.search_params = search_params
        self.net_scale = search_params.net_scale
        self.num_layers = self.net_scale.num_layers
        self.output_indices = output_indices
        self.inplanes = self.net_scale.chs[-1]
        self.logger = logging.getLogger()

        self.primitives_reduce = search_params.primitives_reduce
        self.primitives_normal = search_params.primitives_normal
        self.primitives_for_head = [op for op in search_params.primitives_reduce 
                                    if 'e3' not in op]
        self._initialize_alphas()

        self.blocks = nn.ModuleList()

        self.blocks.append(nn.Sequential(
                        nn.Conv2d(3, self.net_scale.chs[0], 3, 
                                self.net_scale.strides[0], 
                                padding=1,bias=False),
                        nn.BatchNorm2d(self.net_scale.chs[0], 
                                    affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)))

        self.blocks.append(OPS['k3_e1'](
                                self.net_scale.chs[0], 
                                self.net_scale.chs[1], 
                                self.net_scale.strides[1], 
                                dilation=1, affine=True, track=True))

        C_in = self.net_scale.chs[1]
        for i in range(len(self.num_layers)):
            C_out = self.net_scale.chs[i+2]
            stride = self.net_scale.strides[i+2]
            if i+2 == self.output_indices[0]:
                connect_head = True
                prim_reduce = self.primitives_for_head
            else:
                connect_head = False
                prim_reduce = self.primitives_reduce
            self.blocks.append(Block(C_in, C_out, stride, 
                                    self.num_layers[i], 
                                    prim_reduce, 
                                    self.primitives_normal,
                                    [search_params.affine, search_params.track],
                                    connect_head))
            C_in = C_out

        self.blocks.append(Conv1_1(self.net_scale.chs[-2], self.inplanes))
        self.extra = self._make_extra_layers(self.extra_setting[input_size])


    def _initialize_alphas(self):
        num_ops_normal = len(self.primitives_normal)
        num_ops_reduce = len(self.primitives_reduce)
        num_ops_reduce_head = len(self.primitives_for_head)
        self.alphas_normal = nn.ParameterList()
        self.alphas_reduce = nn.ParameterList()
        for i, num_layer in enumerate(self.num_layers):
            if i+2 == self.output_indices[0]:
                num_reduce = num_ops_reduce_head
            else:
                num_reduce = num_ops_reduce
            self.alphas_reduce.append(nn.Parameter(
                1e-3 * torch.randn(1, num_reduce).cuda(), 
                requires_grad=True))
            self.alphas_normal.append(nn.Parameter(
                1e-3 * torch.randn(num_layer-1, num_ops_normal).cuda(), 
                requires_grad=True))


    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = conv_dw(
                    self.inplanes, outplane, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Sequential(
                    nn.Conv2d(self.inplanes, outplane, k, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(outplane),
                    nn.ReLU6(inplace=True)
                )
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1

        return nn.Sequential(*layers)