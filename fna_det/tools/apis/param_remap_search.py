import re
import torch
import logging
from tools.utils import load_checkpoint


def remap_for_archadapt(load_path, model_dict, seed_num_layers=[]):
    seed_dict = load_checkpoint(load_path, map_location='cpu')
    logging.info('Remapping for architecture adaptation starts!')

    # remapping on the depth level
    depth_mapped_dict = {}
    for k in model_dict.keys():
        if k in seed_dict:
            depth_mapped_dict[k] = seed_dict[k]
        elif 'blocks.1' in k:
            seed_key = re.sub('blocks.1', 'blocks.1.layers.0', k)
            if 'tracked' in seed_key and seed_key not in seed_dict:
                continue
            depth_mapped_dict[k] = seed_dict[seed_key]
        elif 'blocks.' in k and 'layers.' in k:
            block_id = int(k.split('.')[1])
            layer_id = int(k.split('.')[3])
            seed_layer_id = seed_num_layers[block_id]-1
            seed_key = re.sub('layers.'+str(layer_id), 
                            'layers.'+str(min(seed_layer_id, layer_id)), 
                            k[:18]) + k[29:]
            if 'tracked' in seed_key and seed_key not in seed_dict:
                continue
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
    logging.info('Remapping for architecture adaptation finished!')
    return model_dict
