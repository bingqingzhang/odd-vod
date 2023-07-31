from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.api_wrappers import COCO

from .coco_video_dataset import CocoVideoDataset
from .imagenet_vid_dataset import ImagenetVIDDataset
from .parsers import CocoVID

import random
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from odd.core import eval_mot
from odd.utils import get_root_logger

@DATASETS.register_module()
class ImagenetVIDODDTrain(ImagenetVIDDataset):
    """ImageNet VID dataset for video object detection."""

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_gt_quality(self):
        gt_quality = []
        for cur_data_info in self.data_infos:
            gt_quality.append(cur_data_info['quality'])
        return gt_quality