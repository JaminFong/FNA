from mmdet.models.detectors import SingleStageDetector

from mmdet.models.registry import DETECTORS
from mmdet.core import bbox2result


@DETECTORS.register_module
class SingleStageDetectorSearch(SingleStageDetector):
    def __init__(self, **kwargs):
        super(SingleStageDetectorSearch, self).__init__(**kwargs)
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs, net_sub_obj = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses, net_sub_obj
