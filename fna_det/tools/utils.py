import logging
import os
import os.path as osp
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils import model_zoo
import torch.distributed as dist

import mmcv
from mmcv.runner.utils import get_dist_info
from tools.multadds_count import comp_multadds_fw


def set_data_path(data_root, data_cfg):
    def add_root(root, path):
        return osp.join(root, path)

    for _, v in data_cfg.items():
        if isinstance(v, dict):
            if hasattr(v, 'ann_file'):
                v.ann_file = add_root(data_root, v.ann_file) 
            if hasattr(v, 'img_prefix'):
                v.img_prefix = add_root(data_root, v.img_prefix)    

            for _, z in v.items():
                if hasattr(z, 'ann_file'):
                    z.ann_file = add_root(data_root, z.ann_file) 
                if hasattr(z, 'img_prefix'):
                    z.img_prefix = add_root(data_root, z.img_prefix)
            

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6


def load_checkpoint(filename,
                    model=None,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if logger is None:
        logger = logging.getLogger()
    # load checkpoint from modelzoo or file or url
    logger.info('Start loading the model from ' + filename)
    if filename.startswith(('http://', 'https://')):
        url = filename
        filename = '../' + url.split('/')[-1]
        if get_dist_info()[0]==0:
            if osp.isfile(filename):
                os.system('rm '+filename)
            os.system('wget -N -q -P ../ ' + url)
        dist.barrier()
    elif filename.startswith(('hdfs://',)):
        url = filename
        filename = '../' + url.split('/')[-1]
        if get_dist_info()[0]==0:
            if osp.isfile(filename):
                os.system('rm '+filename)
            os.system('hdfs dfs -get ' + url + ' ../')
        dist.barrier()
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    if model is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)
        logger.info('Loading the model finished!')
    return state_dict


def parse_net_config(net_config):
    if isinstance(net_config, list):
        return net_config
    elif isinstance(net_config, str):
        str_configs = net_config.split('|')
        return [eval(str_config) for str_config in str_configs]
    else:
        raise TypeError

def load_net_config(path):
    with open(path, 'r') as f:
        net_config = ''
        while True:
            line = f.readline().strip()
            if line:
                net_config += line
            else:
                break
    return net_config


def sort_net_config(net_config):
    def sort_skip(op_list):
        # put the skip operation to the end of one stage
        num = op_list.count('skip')
        for _ in range(num):
            op_list.remove('skip')
            op_list.append('skip')
        return op_list

    sorted_net_config = []
    for cfg in net_config:
        cfg[1] = sort_skip(cfg[1])
        sorted_net_config.append(cfg)
    return sorted_net_config


def get_output_chs(net_config):
    if '|' in net_config:
        net_config = parse_net_config(net_config) 
    stage_chs = []
    for block in net_config[:-1]:
        if block[-1]==2:
            stage_chs.append(block[0][0])
    
    stage_chs.append(net_config[-1][0][1])

    return stage_chs[-4:]


def get_network_madds(backbone, neck, head, input_size, logger, search=False):
    input_data = torch.randn((2,3,)+input_size).cuda()
    backbone_madds, backbone_data = comp_multadds_fw(backbone, input_data)
    backbone_params = count_parameters_in_MB(backbone)
    if neck is not None:
        neck_madds, neck_data = comp_multadds_fw(neck, backbone_data)
        neck_params = count_parameters_in_MB(neck)
    else:
        neck_madds = 0.
        neck_params = 0.
        neck_data = backbone_data
    if hasattr(head, 'search') and search:
        head.search = False
    head_madds, _ = comp_multadds_fw(head, neck_data)
    head_params = count_parameters_in_MB(head)
    if hasattr(head, 'search') and search:
        head.search = True
    total_madds = backbone_madds + neck_madds + head_madds
    total_params = backbone_params + neck_params + head_params

    logger.info("Derived Mult-Adds: [Backbone] %.2fGB [Neck] %.2fGB [Head] %.2fGB [Total] %.2fGB", 
                    backbone_madds/1e3, neck_madds/1e3, head_madds/1e3, total_madds/1e3)
    logger.info("Derived Num Params: [Backbone] %.2fMB [Neck] %.2fMB [Head] %.2fMB [Total] %.2fMB", 
                    backbone_params, neck_params, head_params, total_params)


def convert_sync_batchnorm(module, process_group=None):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.SyncBatchNorm(module.num_features,
                                                module.eps, module.momentum,
                                                module.affine,
                                                module.track_running_stats,
                                                process_group)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output._specify_ddp_gpu_num(1)
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, process_group))
    del module
    return module_output


def init_logger(log_dir=None, level=logging.INFO):
    """Init the logger.

    Args:
        log_dir(str, optional): Log file directory. If not specified, no
            log file will be used.
        level (int or str): See the built-in python logging module.

    Returns:
        :obj:`~logging.Logger`: Python logger.
    """
    rank, _ = get_dist_info()
    logging.basicConfig(
        format='%(asctime)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    if log_dir and rank == 0:
        filename = '{}.log'.format(time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        log_file = osp.join(log_dir, filename)
        _add_file_handler(logger, log_file, level=level)
    return logger


def get_root_logger(log_dir=None, log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level=log_level,
            datefmt='%m/%d %I:%M:%S %p')
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')

    if log_dir and rank == 0:
        filename = '{}.log'.format(time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        log_file = osp.join(log_dir, filename)
        _add_file_handler(logger, log_file, level=log_level)
    return logger


def _add_file_handler(logger,
                    filename=None,
                    mode='w',
                    level=logging.INFO):
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def create_work_dir(work_dir):
    if mmcv.is_str(work_dir):
        work_dir = osp.abspath(work_dir)
        mmcv.mkdir_or_exist(work_dir)
    elif work_dir is None:
        work_dir = None
    else:
        raise TypeError('"work_dir" must be a str or None')
