import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init
from mmdet.models.registry import BACKBONES
from tools.apis.param_remap_search import remap_for_archadapt
from tools.multadds_count import comp_multadds_fw

from .operations import OPS


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, dilation, primitives, affine=False, track=False):
        super(MixedOp, self).__init__()

        self._ops = nn.ModuleList()
        self.primitives = primitives
        for primitive in primitives:
            op = OPS[primitive](C_in, C_out, stride, dilation, affine, track)
            self._ops.add_module('{}'.format(primitive), op)

    def forward(self, x, weights, branch_indices, mixed_sub_obj):
        op_weights = torch.stack([weights[branch_index] for branch_index in branch_indices])
        op_weights = F.softmax(op_weights, dim=-1)
        return sum(op_weight * getattr(self._ops, self.primitives[branch_index])(x) for branch_index, op_weight in zip(
                    branch_indices, op_weights)), \
                sum(op_weight * mixed_sub_obj[branch_index] for branch_index, op_weight in zip(
                    branch_indices, op_weights))


class Block(nn.Module):
    def __init__(self, C_in, C_out, stride, num_layer, search_params, bn_params=[True, True]):
        super(Block, self).__init__()
        self.layers = nn.ModuleList() 
        for inner_idx in range(num_layer):
            if inner_idx == 0:
                self.layers.append(MixedOp(C_in, C_out, stride, 1, 
                                            search_params.primitives_reduce, 
                                            affine=bn_params[0], track=bn_params[1]))
            else:
                self.layers.append(MixedOp(C_out, C_out, 1, 1, 
                                            search_params.primitives_normal, 
                                            affine=bn_params[0], track=bn_params[1]))

    def forward(self, x, weights_normal, weights_reduce, branch_index, block_sub_obj):
        weights = []
        for weight in weights_reduce:
            weights.append(weight)
        for weight in weights_normal:
            weights.append(weight)

        count_sub_obj = []
        for layer, weight, branch_idx, layer_sub_obj in zip(
                    self.layers, weights, branch_index, block_sub_obj):
            x, sub_obj = layer(x, weight, branch_idx, layer_sub_obj)
            count_sub_obj.append(sub_obj)
        return x, sum(count_sub_obj)


class BaseBackbone(nn.Module):
    def __init__(self, search_params, output_indices=[2, 3, 5, 7]):
        super(BaseBackbone, self).__init__()
        self.output_indices = output_indices
        self.search_params = search_params
        self.net_scale = search_params.net_scale
        self.num_layers = self.net_scale.num_layers
        self.logger = logging.getLogger()

        self.primitives_reduce = search_params.primitives_reduce
        self.primitives_normal = search_params.primitives_normal
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
        for n_idx in range(len(self.num_layers)):
            C_out = self.net_scale.chs[n_idx+2]
            stride = self.net_scale.strides[n_idx+2]
            self.blocks.append(Block(C_in, C_out, stride, 
                                    self.num_layers[n_idx], 
                                    search_params,
                                    [search_params.affine, search_params.track]))
            C_in = C_out


    def _initialize_alphas(self):
        num_ops_normal = len(self.search_params.primitives_normal)
        num_ops_reduce = len(self.search_params.primitives_reduce)
        self.alphas_normal = nn.ParameterList()
        self.alphas_reduce = nn.ParameterList()
        for num_layer in self.num_layers:
            self.alphas_reduce.append(nn.Parameter(
                1e-3 * torch.randn(1, num_ops_reduce).cuda(), 
                requires_grad=True))
            self.alphas_normal.append(nn.Parameter(
                1e-3 * torch.randn(num_layer-1, num_ops_normal).cuda(), 
                requires_grad=True))


    @property
    def arch_parameters(self):
        arch_params = nn.ParameterList()
        arch_params.extend(self.alphas_reduce)
        arch_params.extend(self.alphas_normal)
        return arch_params


    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None and m.bias is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        if pretrained.use_load:
            model_dict = remap_for_archadapt(pretrained.load_path, self.state_dict(), pretrained.seed_num_layers)
            self.load_state_dict(model_dict)
    

    def display_arch_params(self, show_arch_params=True):
        alpha_weights = []
        len_arch_params = len(self.arch_parameters)
        alpha_reduces, alpha_normals = self.arch_parameters[:len_arch_params//2], self.arch_parameters[len_arch_params//2:]

        for alpha_reduce, alpha_normal in zip(alpha_reduces,alpha_normals):
            alpha_weights.append(F.softmax(alpha_reduce, dim=-1))
            alpha_weights.append(F.softmax(alpha_normal, dim=-1))

        self.logger.info('alpha_weights \n' + '\n'.join(map(str, alpha_weights)))

        alpha_weights = []
        for alpha_reduce, alpha_normal in zip(alpha_reduces,alpha_normals):
            alpha_weights.extend(F.softmax(alpha_reduce, dim=-1))
            alpha_weights.extend(F.softmax(alpha_normal, dim=-1))

        return [x.tolist() for x in alpha_weights]


    def sample_branch(self, sample_num, training=True, search_stage=0):
        r"""
        input: sample_num
        output: sampled params
        """
        def sample(param, weight, sample_num, training, sample_policy='prob'):
            assert param.shape == weight.shape
            assert sample_policy in ['prob', 'uniform', 'all']

            if len(weight) == 0:
                return torch.tensor([])
            if sample_num == -1:
                sample_policy = 'all'

            if sample_policy == 'prob':
                sampled_index = torch.multinomial(weight, num_samples=sample_num, replacement=False)
            elif sample_policy == 'uniform':
                weight = torch.ones_like(weight)
                sampled_index = torch.multinomial(weight, num_samples=sample_num, replacement=False)
            else:
                sampled_index = torch.arange(start=0, end=weight.shape[-1], step=1, device=weight.device).repeat(param.shape[0], 1)

            sampled_index, _ = torch.sort(sampled_index, descending=False)

            return sampled_index
        len_arch_params = len(self.arch_parameters)
        params_reduces, params_normals = self.arch_parameters[:len_arch_params//2], self.arch_parameters[len_arch_params//2:]

        weights_reduces = []
        weights_normals = []

        sampled_indices_reduces = []
        sampled_indices_normals = []
        for param in params_reduces:
            weights_reduces.append(F.softmax(param, dim=-1))
        for param in params_normals:
            weights_normals.append(F.softmax(param, dim=-1))

        sample_policy = self.search_params.sample_policy if search_stage == 1 else 'uniform'
        for param, weight in zip(params_reduces, weights_reduces): 
            sampled_index = sample(param, weight, sample_num, training, sample_policy)
            sampled_indices_reduces.append(sampled_index)
        for param, weight in zip(params_normals, weights_normals): 
            sampled_index = sample(param, weight, sample_num, training, sample_policy)
            sampled_indices_normals.append(sampled_index)

        sampled_indices = []
        for sampled_indices_reduce, sampled_indices_normal in zip(sampled_indices_reduces, sampled_indices_normals):
            index = []
            for idx in sampled_indices_reduce:
                index.append(idx)
            for idx in sampled_indices_normal:
                index.append(idx)
            sampled_indices.append(index)
        self.alpha_index = sampled_indices
        return sampled_indices


    def get_sub_obj_list(self, sub_obj_cfg, data_shape):
        if sub_obj_cfg.type=='flops':
            flops_list_sorted = self.get_flops_list(data_shape)
            self.sub_obj_list = copy.deepcopy(flops_list_sorted)


    def get_flops_list(self, input_shape):
        data = torch.randn(input_shape)
        block_flops = []
        data = self.blocks[0](data)
        data = self.blocks[1](data)

        for block in self.blocks[2:]:
            layer_flops = []
            if hasattr(block, 'layers'):
                for layer in block.layers:
                    op_flops = []
                    for op in layer._ops:
                        flops, op_data = comp_multadds_fw(op, data, 'B', 'cpu')
                        op_flops.append(flops)
                    data = op_data
                    layer_flops.append(op_flops)
                block_flops.append(layer_flops)
        return block_flops


    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
        # if mode and self.freeze_bn:
        #     for m in self.modules():
        #         # trick: eval have effect on BatchNorm only
        #         if isinstance(m, BatchNorm):
        #             m.eval()
