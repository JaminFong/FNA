import re
import torch
import logging
from collections import OrderedDict


def remap_for_paramadapt(load_path, model_dict, seed_num_layers=[]):
    seed_dict = load_checkpoint(load_path, map_location='cpu')
    logging.info('Remapping for parameter adaptation starts!')

    # remapping on the depth level
    depth_mapped_dict ={}
    for k in model_dict.keys():
        if k in seed_dict:
            depth_mapped_dict[k] = seed_dict[k]
        elif 'blocks.' in k and 'layers.' in k:
            block_id = int(k.split('.')[1])
            layer_id = int(k.split('.')[3])
            seed_layer_id = seed_num_layers[block_id]-1
            seed_key = re.sub('layers.'+str(layer_id), 'layers.'+str(seed_layer_id), k)
            depth_mapped_dict[k] = seed_dict[seed_key]

    # remapping on the width and kernel level simultaneously
    mapped_dict = {}
    for k, v in depth_mapped_dict.items():
        if k in model_dict:
            if ('weight' in k) & (len(v.size()) != 1):
                output_dim = min(model_dict[k].size()[0], v.size()[0])
                input_dim = min(model_dict[k].size()[1], v.size()[1])
                w_model = model_dict[k].size()[2]
                w_pre = v.size()[2]
                h_model = model_dict[k].size()[3]
                h_pre = v.size()[3]
                w_min = min(model_dict[k].size()[2], v.size()[2])
                h_min = min(model_dict[k].size()[3], v.size()[3])

                mapped_dict[k] = torch.zeros_like(model_dict[k], requires_grad=True)
                mapped_dict[k].narrow(0, 0, output_dim).narrow(1, 0, input_dim).narrow(
                    2, (w_model - w_min) // 2, w_min).narrow(3, (h_model - h_min) // 2, h_min).copy_(
                    v.narrow(0, 0, output_dim).narrow(1, 0, input_dim).narrow(
                    2, (w_pre - w_min) // 2, w_min).narrow(3, (h_pre - h_min) // 2, h_min))

            elif len(v.size()) != 0:
                param_dim = min(model_dict[k].size()[0], v.size()[0])
                mapped_dict[k] = model_dict[k]
                mapped_dict[k].narrow(0, 0, param_dim).copy_(v.narrow(0, 0, param_dim))
            else:
                mapped_dict[k] = v
    
    model_dict.update(mapped_dict)
    logging.info('Remapping for parameter adaptation finished!')
    return model_dict

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
    # """
    # if logger is None:
    #     logger = logging.getLogger()
    # # load checkpoint from modelzoo or file or url
    # logger.info('Start loading the model from ' + filename)
    # if filename.startswith(('http://', 'https://')):
    #     url = filename
    #     filename = '../' + url.split('/')[-1]
    #     if get_dist_info()[0]==0:
    #         if osp.isfile(filename):
    #             os.system('rm '+filename)
    #         os.system('wget -N -q -P ../ ' + url)
    #     dist.barrier()
    # else:
    #     if not osp.isfile(filename):
    #         raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename[0], map_location=map_location)
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