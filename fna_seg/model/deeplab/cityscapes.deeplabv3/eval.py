#!/usr/bin/env python3
# encoding: utf-8
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from config import config
from datasets.cityscapes.cityscapes import Cityscapes
from engine.evaluator import Evaluator
from engine.logger import get_logger
from network import DeepLabV3
from seg_opr.metric import compute_score, hist_info
from utils.config_utils import load_net_config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img

logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        pred = self.sliding_eval(img, config.eval_crop_size,
                                 config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes,
                                                       pred,
                                                       label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        if self.save_path is not None:
            fn = name + '_color.png'
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imwrite(os.path.join(self.save_path, fn), comp_img)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(), True)
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--net_config', default=None, type=str, help='the path of net_config')
    parser.add_argument('--data_path', default=None, type=str, help='the path of datapath')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    if args.net_config is not None:
        config.net_config = load_net_config(args.net_config)

    network = DeepLabV3(is_training=False, config=config)
    if args.data_path is not None:
        config.dataset_path = args.data_path
    data_setting = {'img_root': config.dataset_path,
                    'gt_root': config.dataset_path,
                    'train_source': osp.join(config.dataset_path, config.train_source),
                    'eval_source': osp.join(config.dataset_path, config.eval_source)}
    dataset = Cityscapes(data_setting, 'val', None)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(args.checkpoint, 'last', config.val_log_file,
                      config.link_val_log_file)
