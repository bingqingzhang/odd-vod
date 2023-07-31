# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F
from addict import Dict
import torch.nn as nn
from mmdet.models import build_detector

from .base import BaseVideoDetector
from ..builder import MODELS


@MODELS.register_module()
class FasterrcnnOdd(BaseVideoDetector):

    def __init__(self,
                 detector,
                 pretrains=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(FasterrcnnOdd, self).__init__(init_cfg)
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
        for param in self.detector.parameters():
            param.requires_grad = False
        self.iqa_module = self.build_iqa()
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.replace_relu = nn.ReLU(inplace=True)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def build_iqa(self):
        return nn.Sequential(
            nn.AdaptiveMaxPool2d(7),
        )

    # def forward(self,
    #             mode,
    #             **kwargs):
    #     if mode == 'train':
    #         return self.forward_train(**kwargs)
    #     else:
    #         return self.forward_test(**kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      quality=torch.tensor(10),
                      **kwargs):
        assert quality.max() <= 1, 'quality should be less than 1'
        bath_num = len(img)
        quality = quality.reshape(bath_num, -1).to(torch.float32)
        all_x = self.detector.extract_feat(img)

        losses = dict()
        res = self.iqa_module(all_x[0])
        res = res.view(res.size(0), -1)
        res = self.fc1(res)
        self.replace_relu(res)
        res = self.fc2(res)
        res = 10 * F.sigmoid(res)
        losses['loss_iqa'] = F.smooth_l1_loss(res, 10 * quality)
        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    out_result_path=None,
                    out_result_flag=None,
                    **kwargs):
        if out_result_flag and out_result_path:
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
            return results
        x = self.detector.extract_feat(img)
        res = self.iqa_module(x[0])
        res = res.view(res.size(0), -1)
        res = self.fc1(res)
        self.replace_relu(res)
        res = self.fc2(res)
        res = F.sigmoid(res)
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
        results['iqa_reg'] = res
        results['det_bboxes'] = outs[0]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
