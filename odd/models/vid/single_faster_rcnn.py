# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from addict import Dict
from mmdet.models import build_detector

from .base import BaseVideoDetector
from ..builder import MODELS


@MODELS.register_module()
class SingleFasterRcnn(BaseVideoDetector):

    def __init__(self,
                 detector,
                 pretrains=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SingleFasterRcnn, self).__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            detector_pretrain = pretrains.get('detector', None)
            if detector_pretrain:
                detector.init_cfg = dict(
                    type='Pretrained', checkpoint=detector_pretrain)
            else:
                detector.init_cfg = None
        self.detector = build_detector(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'vanilla-faster-rcnn video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """

        """
        assert len(img) == 1, \
            'vanilla faster-rcnn video detector only supports 1 batch size per gpu for now.'

        # all_imgs = torch.cat((img, ref_img[0]), dim=0)
        # all_x = self.detector.extract_feat(all_imgs)
        all_x = self.detector.extract_feat(img)
        x = []
        x.append(all_x[0])

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.detector.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes,
            gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        x = self.detector.extract_feat(img)
        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
        else:
            proposal_list = proposals
        outs = self.detector.roi_head.simple_test(
            x,
            proposal_list,
            img_metas,
            rescale=rescale)
        results = dict()
        results['det_bboxes'] = outs[0]
        if len(outs) == 2:
            results['det_masks'] = outs[1]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
