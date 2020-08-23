import torch
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.test_mixins import RPNTestMixin, MaskTestMixin
from mmdet.models import builder
from mmdet.models.registry import DETECTORS
from mmdet.core import bbox2result, bbox2roi, delta2bbox, bbox_overlaps
import numpy as np


@DETECTORS.register_module
class CaSe(BaseDetector, RPNTestMixin, MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 count_head=None,
                 similar_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CaSe, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if count_head is not None:
            self.count_head = builder.build_head(count_head)

        if similar_head is not None:
            self.similar_head = builder.build_head(similar_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
        proposals = self.rpn_head.get_bboxes(*proposal_inputs)

        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        if self.test_cfg.rcnn.nms.type == 'cas_nms':
            dets = delta2bbox(rois[:, 1:], bbox_pred, self.bbox_head.target_means, self.bbox_head.target_stds)
            rois_count = bbox2roi([dets[:, 4:]])
            roi_feats_count = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois_count)
            count_pred = self.count_head(roi_feats_count)
            selected_ids_H = torch.nonzero((count_pred >= self.test_cfg.rcnn.nms.t_1).squeeze())
            similarity_matrix = count_pred.new_zeros(count_pred.size(0), count_pred.size(0))
            overlaps = bbox_overlaps(rois_count[:, 1:], rois_count[:, 1:])
            overlap_pair_id = torch.nonzero((overlaps >= self.test_cfg.rcnn.nms.thresh))
            overlap_pair_id = overlap_pair_id[overlap_pair_id[:, 0] != overlap_pair_id[:, 1], :]
            if selected_ids_H.size(0) > 0:
                for si in range(selected_ids_H.size(0)):
                    i = selected_ids_H[si].squeeze().int().long()
                    selected_ids = overlap_pair_id[overlap_pair_id[:, 0] == i, :]
                    selected_ids_co = selected_ids[count_pred[selected_ids[:, 1], 0] >= self.test_cfg.rcnn.nms.t_2, 1]
                    if selected_ids_co.size(0) > 0:
                        sim_feats_H = self.similar_head(roi_feats_count[i][None, :])
                        if selected_ids_co.size(0) > 1:
                            sim_feats_co = self.similar_head(roi_feats_count[selected_ids_co])
                        else:
                            sim_feats_co = self.similar_head(roi_feats_count[selected_ids_co][None, :])
                        similarity = F.pairwise_distance(sim_feats_H, sim_feats_co, p=2)
                        similarity_matrix[selected_ids_co, i] = similarity

            det_bboxes, det_labels = self.bbox_head.get_det_bboxes_cas_nms(rois, cls_score, bbox_pred, count_pred,
                                                                           similarity_matrix,
                                                                           img_shape, scale_factor, rescale=rescale,
                                                                           cfg=self.test_cfg.rcnn)

        else:
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=self.test_cfg.rcnn)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        # if self.with_mask:
        #     segm_results = self.simple_test_mask(
        #         x, img_meta, det_bboxes, det_labels, rescale=rescale)
        return bbox_results
