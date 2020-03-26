import torch

from mmcv.runner import OptimizerHook
from mmdet.core.utils.dist_utils import allreduce_grads

from collections import OrderedDict
import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _unflatten_dense_tensors,
                          _take_tensors)

class ArchOptimizerHook(OptimizerHook):

    def after_train_iter(self, runner):
        runner.arch_optimizer.zero_grad()
        if runner.sub_obj_cfg.if_sub_obj:
            loss_sub_obj = torch.log(runner.outputs['sub_obj']) / \
                    torch.log(torch.tensor(runner.sub_obj_cfg.log_base))
            runner.outputs['loss'] += loss_sub_obj * runner.sub_obj_cfg.sub_loss_factor
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.arch_optimizer.step()
        self.rescale_arch_params(runner.super_backbone)


    def rescale_arch_params(self, model):
        """
            rescale the architecture parameters 
            that is to add the rescale_value (bias) to the updated architecture parameters 
            to maintain the magnitude of the softmax outputs of non-updated params
        """

        def comp_rescale_value(old_weights, new_weights, index, block_id, branch_id):
            old_exp_sum = old_weights.exp().sum()
            new_drop_arch_params = [new_weights[block_id][branch_id][h_idx
                                                        ] for h_idx in index]
            new_exp_sum = torch.stack(new_drop_arch_params).exp().sum()
            rescale_value = torch.log(old_exp_sum / new_exp_sum)

            return rescale_value

        if hasattr(model, 'module'):
            model = model.module
        
        alpha_head_index = model.alpha_head_index
        alpha_head_weights_drop = model.alpha_head_weights_drop
        alpha_stack_index = model.alpha_stack_index
        alpha_stack_weights_drop = model.alpha_stack_weights_drop

        # rescale the arch params for head layers
        for i, (alpha_head_weights_drop_block, alpha_head_index_block) in enumerate(
                                            zip(alpha_head_weights_drop, alpha_head_index)):
            for j, (alpha_head_weights_drop_branch, alpha_head_index_branch) in enumerate(
                                zip(alpha_head_weights_drop_block, alpha_head_index_block)):
                rescale_value = comp_rescale_value(alpha_head_weights_drop_branch,
                                                    model.alpha_head_weights,
                                                    alpha_head_index_branch, i, j)
                for idx in alpha_head_index_branch:
                    model.alpha_head_weights[i].data[j][idx] += rescale_value

        # rescale the arch params for stack layers
        for i, (alpha_stack_weights_drop_block, alpha_stack_index_block) in enumerate(
                                            zip(alpha_stack_weights_drop, alpha_stack_index)):
            for j, (alpha_stack_weights_drop_branch, alpha_stack_index_branch) in enumerate(
                                zip(alpha_stack_weights_drop_block, alpha_stack_index_block)):
                rescale_value = comp_rescale_value(alpha_stack_weights_drop_branch,
                                                    model.alpha_stack_weights,
                                                    alpha_stack_index_branch, i, j)
                for idx in alpha_stack_index_branch:
                    model.alpha_stack_weights[i].data[j][idx] += rescale_value


class ArchDistOptimizerHook(ArchOptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb


    def after_train_iter(self, runner):
        runner.arch_optimizer.zero_grad()
        if runner.sub_obj_cfg.if_sub_obj:
            loss_sub_obj = torch.log(runner.outputs['sub_obj']) / \
                    torch.log(torch.tensor(runner.sub_obj_cfg.log_base))
            runner.outputs['loss'] += loss_sub_obj * runner.sub_obj_cfg.sub_loss_factor
        runner.outputs['loss'].backward()
        allreduce_grads(runner.model, self.coalesce, self.bucket_size_mb)
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.arch_optimizer.step()
        # self.rescale_arch_params(runner.super_backbone)

