import copy
import os
import os.path as osp
import time

import torch
import torch.distributed as dist

import mmcv
from mmcv.runner.checkpoint import load_checkpoint
from mmcv.runner import Runner, hooks
from mmcv.runner.hooks import (CheckpointHook, Hook, IterTimerHook,
                               LrUpdaterHook, OptimizerHook, lr_updater)
from mmcv.runner.priority import get_priority
from mmcv.runner.utils import (get_dist_info, get_host_info, get_time_str,
                               obj_from_dict)
from models.derive_arch import ArchGenerate_FNA
from models.derived_retinanet_backbone import FNA_Retinanet
from models.derived_ssdlite_backbone import FNA_SSDLite
from tools.hooks.fna_search_hooks import (DropProcessHook, ModelInfoHook,
                                          NASTextLoggerHook)
from tools.hooks.optimizer import ArchOptimizerHook


class NASRunner(Runner):
    def __init__(self, *args, **kwargs):
        assert 'cfg' in kwargs.keys()
        self.cfg = kwargs.pop('cfg')
        super(NASRunner, self).__init__(*args, **kwargs)
        
        self.sub_obj_cfg = self.cfg.sub_obj
        self.type = self.cfg.type
        super_backbone = self.model.module.backbone if hasattr(self.model, 'module') \
                                                        else self.model.backbone
        self.arch_gener = ArchGenerate_FNA(super_backbone)
        if self.cfg.type == 'Retinanet':
            self.der_Net = lambda net_config: FNA_Retinanet(net_config)
        elif self.cfg.type == 'SSDLite':
            self.der_Net = lambda net_config: FNA_SSDLite(self.cfg.input_size, 
                                net_config, self.cfg.model.backbone.output_indices)
        else:
            raise NotImplementedError

        nas_optimizers = self.cfg.optimizer
        self.optimizer, self.arch_optimizer = self.init_nas_optimizer(nas_optimizers)
        self._arch_hooks = []


    def run(self, data_loaders, workflow, max_epochs, arch_update_epoch, **kwargs):
        if self.cfg.alter_type=='epoch':
            self.run_epoch_alter(data_loaders, workflow, max_epochs, arch_update_epoch, **kwargs)
        elif self.cfg.alter_type=='step':
            self.run_step_alter(data_loaders, workflow, max_epochs, arch_update_epoch, **kwargs)
        else:
            raise TypeError('The alternation type of optimization must be epoch or step')


    def run_epoch_alter(self, data_loaders, workflow, max_epochs, arch_update_epoch, **kwargs):
        """Start running. Arch and weight optimization alternates by epoch.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run', 'train')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    assert mode in ['train', 'arch', 'val']
                    if mode in ['train', 'arch']:
                        epoch_runner = getattr(self, 'train'+'_epoch_alter')
                    else:
                        epoch_runner = getattr(self, 'val')
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    elif mode in ['arch', 'val'] and self.epoch < arch_update_epoch:
                        break
                    data_loader = data_loaders[0] if mode=='train' else data_loaders[1]
                    epoch_runner(data_loader, mode=mode)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run', 'train')


    def train_epoch_alter(self, data_loader, **kwargs):
        """
            Arch and weight optimization alternates by epoch.
        """
        mode = kwargs.pop('mode')
        self.mode=mode
        self.model.train()
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch', mode)
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter', mode)
            outputs = self.batch_processor(
                self.model, data_batch, mode=mode, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter', mode)
            self._iter += 1

        self.call_hook('after_train_epoch', mode)
        if mode == 'train':
            self._epoch += 1


    def run_step_alter(self, data_loaders, workflow, max_epochs, arch_update_epoch, **kwargs):
        """Start running. Arch and weight optimization alternates by step.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run', 'train')

        while self.epoch < max_epochs:
            self.search_stage = 0 if self.epoch<self.cfg.arch_update_epoch else 1
            self.train_step_alter(data_loaders)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run', 'train')

    def set_param_grad_state(self, stage):
        def set_grad_state(params, state):
            for group in params:
                for param in group['params']:
                    param.requires_grad_(state)
        if stage == 'arch':
            state_list = [True, False] # [arch, weight]
        elif stage == 'train':
            state_list = [False, True]
        else:
            state_list = [False, False]
        set_grad_state(self.arch_optimizer.param_groups, state_list[0])
        set_grad_state(self.optimizer.param_groups, state_list[1])

    def train_step_alter(self, data_loaders, **kwargs):
        """
            Arch and weight optimization alternates by step.
        """
        self.model.train()
        self._max_iters = self._max_epochs * len(data_loaders[0])
        self._inner_iter = 0
        data_loader_iters = []
        for data_loader in data_loaders:
            data_loader_iters.append(iter(data_loader))
        self.data_loader = data_loaders[0]
        self.call_hook('before_train_epoch', 'train')
        self.call_hook('before_train_epoch', 'arch')
        while self._inner_iter < len(data_loaders[0]):
            for i, flow in enumerate(self.cfg.workflow):
                mode, steps = flow
                self.mode = mode
                self.set_param_grad_state(mode)

                if mode=='arch' and self.search_stage == 0:
                    continue
                for _ in range(steps):
                    try:
                        data_batch = data_loader_iters[i].next()
                    except:
                        data_loader_iters[i] = iter(data_loaders[i])
                        data_batch = data_loader_iters[i].next()
                    self.call_hook('before_train_iter', mode)
                    outputs = self.batch_processor(
                        self.model, data_batch, mode=mode, 
                        search_stage=self.search_stage, 
                        net_type=self.cfg.type, **kwargs)
                    if not isinstance(outputs, dict):
                        raise TypeError('batch_processor() must return a dict')
                    if 'log_vars' in outputs:
                        self.log_buffer.update(outputs['log_vars'],
                                            outputs['num_samples'])
                    self.outputs = outputs
                    if mode=='arch': # TODO: used for 5-step train
                        tmp = self._inner_iter
                        self._inner_iter -= 1
                        self.call_hook('after_train_iter', mode)
                        self._inner_iter = tmp
                    else:
                        self.call_hook('after_train_iter', mode)
                    if mode=='train':
                        self._inner_iter += 1
                        self._iter += 1

        self.call_hook('after_train_epoch', 'train')
        self.call_hook('after_train_epoch', 'arch')
        self._epoch += 1


    def init_nas_optimizer(self, optimizers):
        """
        init nas optimizer: weight optimizer and architecture optimizer
        args:
            dict (weight_optim config & arch_optim config)
        returns:
            tuple(weight_optim, arch_optim)
        """

        if isinstance(optimizers, dict):
            assert hasattr(optimizers, 'weight_optim') and hasattr(optimizers, 'arch_optim')
            optim_list = []
            arch_params_id = list(map(id, self.model.module.backbone.arch_parameters
                if hasattr(self.model, 'module') else self.model.backbone.arch_parameters))
            weight_params = filter(lambda p: id(p) not in arch_params_id, self.model.parameters())
            arch_params = filter(lambda p: id(p) in arch_params_id, self.model.parameters())

            for key, optim in optimizers.items():
                if key == 'weight_optim':
                    params = weight_params
                elif key == 'arch_optim':
                    params = arch_params
                else:
                    assert KeyError
                optimizer = obj_from_dict(optim.optimizer, torch.optim, dict(params=params))
                optim_list.append(optimizer)
        else:
            raise TypeError(
                'optimizer must be a dict, ''but got {}'.format(type(optimizers)))

        return tuple(optim_list)


    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None or self.arch_optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        if self.mode is 'train':
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.mode is 'arch':
            return [group['lr'] for group in self.arch_optimizer.param_groups]
    

    def register_training_hooks(self,
                                lr_config,
                                weight_optim_config=None,
                                arch_optim_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - Weight/Arch_OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if weight_optim_config is None:
            weight_optim_config = {}
        if arch_optim_config is None:
            arch_optim_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}

        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(weight_optim_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        # self.register_hook(ModelInfoHook(self.cfg.model_info_interval), priority='VERY_LOW')
        self.register_hook(DropProcessHook(), priority='LOW')
        self.register_hook(IterTimerHook())

        self.register_arch_hook(self.build_hook(arch_optim_config, ArchOptimizerHook))
        self.register_arch_hook(ModelInfoHook(self.cfg.model_info_interval), priority='VERY_LOW')
        self.register_arch_hook(DropProcessHook(), priority='LOW')
        self.register_arch_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config) # logger_hook for arch_hook will be added inside


    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            if info.type == 'TextLoggerHook':
                logger_hook = NASTextLoggerHook(log_interval)
            else:
                logger_hook = obj_from_dict(
                    info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')
            del logger_hook.priority
            self.register_arch_hook(logger_hook, priority='VERY_LOW')


    def register_arch_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._arch_hooks) - 1, -1, -1):
            if priority >= self._arch_hooks[i].priority:
                self._arch_hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._arch_hooks.insert(0, hook)


    def call_hook(self, fn_name, mode='train'):
        hooks_run = self._hooks if mode=='train' else self._arch_hooks
        for hook in hooks_run:
            getattr(hook, fn_name)(self)


    def load_checkpoint(self, filename, map_location='cpu', strict=True):
        self.logger.info('load checkpoint from %s', filename)

        if filename.startswith(('http://', 'https://')):
            url = filename
            filename = '../' + url.split('/')[-1]
            if get_dist_info()[0]==0:
                if osp.isfile(filename):
                    os.system('rm '+filename)
                os.system('wget -N -q -P ../ ' + url)
            dist.barrier()

        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)
