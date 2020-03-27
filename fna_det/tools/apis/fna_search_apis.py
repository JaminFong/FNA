from __future__ import division

from collections import OrderedDict

import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook
from mmdet.apis.env import get_root_logger
from mmdet.core import (CocoDistEvalRecallHook, CocoDistEvalmAPHook,
                        DistEvalmAPHook, DistOptimizerHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from models.dropped_model import Dropped_Network
from models.dropped_model_ssdlite import SSDLite_Dropped_Network
from tools.apis.fna_search_runner import NASRunner
from tools.hooks.optimizer import ArchDistOptimizerHook
from tools.hooks.eval_hooks import CocoDistEvalmAPHook_


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode=True, mode='train', search_stage=0, net_type=''):
    if train_mode:
        if mode == 'train':
            sample_num = 1
        elif mode == 'arch':
            sample_num = -1
    else:
        sample_num = -1

    if net_type == 'Retinanet':
        DroppedBackBone = Dropped_Network
    elif net_type == 'SSDLite':
        DroppedBackBone = SSDLite_Dropped_Network
    else:
        raise NotImplementedError

    # if sample_num is not None:
    _ = model.module.backbone.sample_branch(sample_num, search_stage=search_stage)

    if hasattr(model, 'module'):
        model.module.backbone = DroppedBackBone(model.module.backbone)
    else:
        model.backbone = DroppedBackBone(model.backbone)

    losses, sub_obj = model(**data)
    sub_obj = torch.mean(sub_obj)
    loss, log_vars = parse_losses(losses)
    log_vars['sub_obj'] = sub_obj.item()
    outputs = dict(
        loss=loss, sub_obj=sub_obj, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def search_detector(model,
                   datasets,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, datasets, cfg, validate=validate, logger=logger)
    else:
        _non_dist_train(model, datasets, cfg, validate=validate, logger=logger)


def _dist_train(model, datasets, cfg, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True) for dataset in datasets
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = NASRunner(model, batch_processor, None, cfg.work_dir, cfg.log_level, cfg=cfg, logger=logger)

    # register hooks
    weight_optim_config = DistOptimizerHook(**cfg.optimizer.weight_optim.optimizer_config)
    arch_optim_config = ArchDistOptimizerHook(**cfg.optimizer.arch_optim.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, weight_optim_config, arch_optim_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
        else:
            if cfg.dataset_type == 'CocoDataset':
                # runner.register_hook(CocoDistEvalmAPHook_(datasets[1]))
                runner.register_hook(CocoDistEvalmAPHook(cfg.data.val_))
            else:
                runner.register_hook(DistEvalmAPHook(cfg.data.val))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, cfg.arch_update_epoch)


def _non_dist_train(model, datasets, cfg, validate=False, logger=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for dataset in datasets
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = NASRunner(model, batch_processor, None, cfg.work_dir, cfg.log_level, cfg=cfg, logger=logger)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer.weight_optim.optimizer_config,
                                    cfg.optimizer.arch_optim.optimizer_config, 
                                    cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, cfg.arch_update_epoch)
