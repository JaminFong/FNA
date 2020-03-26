import copy
import logging
import mmcv
import random

from collections import defaultdict
from mmdet.datasets import CocoDataset, RepeatDataset
from pycocotools.coco import COCO


def build_divide_dataset(cfg, part_1_ratio=0.5, seed=520):
    """Need to change `coco.getImgIds()` output, i.e., 
    `self.imgs` in a COCO instance. 
    `self.imgs` is created by `self.dataset['images']`.
    Thus, hacking `self.dataset['images']` when initializing a 
    COCO class.
    """
    logging.info('Prepare datasets.')
    data_type = cfg.train.type
    if data_type == 'RepeatDataset':
        train_cfg = cfg.train.dataset
    else:
        train_cfg = cfg.train
    assert train_cfg.pop('type') == 'CocoDataset', 'Only support COCO.'
    annotations = mmcv.load(train_cfg.pop('ann_file'))
    images = annotations.pop('images')
    part_1_annotations = copy.copy(annotations)
    part_2_annotations = copy.copy(annotations)

    part_1_length = int(part_1_ratio * len(images))
    if seed is not None:
        random.seed(seed)
        random.shuffle(images)
    part_1_images = images[:part_1_length]
    part_2_images = images[part_1_length:]

    part_1_annotations['images'] = part_1_images
    part_2_annotations['images'] = part_2_images

    part_1_coco = COCOFromDict(part_1_annotations)
    part_2_coco = COCOFromDict(part_2_annotations)
    part_1_dataset = InitDatasetFromCOCOClass(**train_cfg, ann_file=part_1_coco)
    part_2_dataset = InitDatasetFromCOCOClass(**cfg.val, ann_file=part_2_coco)
    if data_type == 'RepeatDataset':
        part_1_dataset = RepeatDataset(part_1_dataset, cfg.train.times)
    logging.info(f'Finished preparing datasets.')

    return part_1_dataset, part_2_dataset


class InitDatasetFromCOCOClass(CocoDataset):

    def load_annotations(self, coco_class):
        """Passing ann_file the coco_class when initializing 
        the class.
        """
        # original loading
        self.coco = coco_class
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos


class COCOFromDict(COCO):

    def __init__(self, anno_dict):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        assert type(anno_dict) == dict
        self.dataset = anno_dict
        self.createIndex()

    def createIndex(self):
        """Re-write this method because I do not wanna see the default printing...
        """
        # Annoying!
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
