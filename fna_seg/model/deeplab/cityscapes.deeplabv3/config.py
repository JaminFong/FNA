# encoding: utf-8

from __future__ import absolute_import, division, print_function

import argparse
import os.path as osp
import sys
import time

import numpy as np
import torch.utils.model_zoo as model_zoo

from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'fna_seg'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.cluster_dir = '/job_data'
middle_dir = C.root_dir
C.log_dir = osp.abspath(osp.join(middle_dir, 'log', C.this_dir))
C.log_dir_link = osp.join(C.abs_dir, 'log')

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""
C.dataset_path = './cityscapes'

C.train_source = "train.lst"
C.eval_source = "val.lst"
C.test_source = "val.lst"
C.is_test = False

"""Path Config"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir, 'furnace'))
from utils.pyt_utils import model_urls

"""Image Config"""
C.num_classes = 19
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.target_size = 769
C.image_height = 769
C.image_width = 769
C.num_train_imgs = 2975
C.num_eval_imgs = 500

""" Settings for network, this would be different for each kind of model"""
C.fix_bias = True
C.fix_bn = False
C.sync_bn = True
# kaiming normal

C.backbone_bn_eps = 1e-5
C.backbone_bn_momentum = 0.1
C.backbone_init = "normal"

C.bn_eps = 1e-5
C.bn_momentum = 0.1
C.init = "normal"

# C.pretrained_model = "./resnet18_v1.pth"

"""Train Config"""
C.lr = 1e-2
C.lr_power = 0.9
C.momentum = 0.9
C.warmup_iters = 5000
C.weight_decay = 5e-4
C.batch_size =  16  # 4 * C.num_gpu
C.nepochs = 100
C.niters_per_epoch = 1000
C.num_workers = 2
C.train_scale_array = [0.75, 1, 1.25, 1.5, 1.75, 2.0]

C.net_config = """[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k3_e6', 'k3_e6', 'skip', 'skip'], 2]|
[[24, 32], ['k7_e6', 'k3_e6', 'k3_e6', 'k3_e6'], 2]|
[[32, 64], ['k7_e6', 'k7_e6', 'k5_e6', 'k7_e6', 'k7_e6', 'k5_e6'], 2]|
[[64, 96], ['k7_e6', 'k7_e6', 'k7_e6', 'k7_e6', 'skip', 'skip'], 1]|
[[96, 160], ['k7_e6', 'k5_e6', 'skip', 'skip'], 2]|
[[160, 320], ['k7_e3'], 1]"""     #fna

"""BACKBONE Config"""

C.width_multi = 1.
# mb  mb1 mb_jm mb_narrow no_pretrain   mb_dila_weights   supernet    mixed   mb_bn   mb_narrow_bn    mb_narrow_std
C.pretrain = "mb_jm"
C.seed_num_layers =[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
C.load_path='seed_mbv2.pt',
# [4,4,1] [4,6,1] [6,6,3]
"""ASPP Config"""
C.MODEL_ASPP_OUTDIM = 256
C.MODEL_OUTPUT_STRIDE = 16

C.ASPP = "Sep"
  #   ResBlock Sep Normal   DPC Nothing CAS

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]  # multi scales: 0.5, 0.75, 1, 1.25, 1.5, 1.75
C.eval_flip = False  # True if use the ms_flip strategy
C.eval_base_size = 769
C.eval_crop_size = 769

"""Display Config"""
C.snapshot_iter = 1000
C.record_info_iter = 20
C.display_iter = 50


def open_tensorboard():
    pass


if __name__ == '__main__':
    print(config.epoch_num)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
