import copy
import datetime
import logging

import torch

from mmcv.runner.hooks import Hook, TextLoggerHook
from mmdet.models.builder import build_neck
from tools.multadds_count import comp_multadds, comp_multadds_fw
from tools.utils import count_parameters_in_MB, get_network_madds


class DropProcessHook(Hook):

    def before_train_iter(self, runner):
        runner.super_backbone = runner.model.module.backbone

    def after_train_iter(self, runner):
        runner.model.module.backbone = runner.super_backbone
        del runner.super_backbone
        

class ModelInfoHook(Hook):
    def __init__(self, interval):
        self.iter = 0
        self.interval = interval

    def after_train_iter(self, runner):
        if self.iter % self.interval == 0 and runner.mode=='arch':
            self.comp_det_madds(runner, show_arch_params=True)
        self.iter += 1

    def after_train_epoch(self, runner):
        self.iter = 0
        runner.logger.info('EPOCH %d finished!'% (runner.epoch + 1))
        self.comp_det_madds(runner)

    def comp_det_madds(self, runner, show_arch_params=True):
        alphas = runner.model.module.backbone.display_arch_params(show_arch_params=show_arch_params)
        derived_archs = runner.arch_gener.derive_archs(alphas, logger=runner.logger)
        derived_model = runner.der_Net(derived_archs)

        runner.logger.info('\n'+derived_archs)
        neck_cfg = copy.deepcopy(runner.cfg.model.neck)

        if neck_cfg is not None:
            neck_model = build_neck(neck_cfg)
        else:
            neck_model = None

        get_network_madds(derived_model, neck_model, runner.model.module.bbox_head, 
                        runner.cfg.image_size_madds, runner.logger, search=True)


class NASTextLoggerHook(TextLoggerHook):

    def _log_info(self, log_dict, runner):
        if runner.mode in ['train', 'arch']:
            log_str = 'Epoch({})[{}][{}/{}] lr: {:.5f}, '.format(
                runner.mode, log_dict['epoch'], log_dict['iter'], len(runner.data_loader),
                log_dict['lr'])
            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += 'eta: {}, '.format(eta_str)
                log_str += ('time: {:.3f}, data_time: {:.3f}, '.format(
                    log_dict['time'], log_dict['data_time']))
                log_str += 'memory: {}, '.format(log_dict['memory'])
        else:
            log_str = 'Epoch({}) [{}][{}]  '.format(
                log_dict['mode'], log_dict['epoch'] - 1, log_dict['iter'])
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)
