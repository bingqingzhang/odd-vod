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
class ImagenetVIDDatasetODDs(ImagenetVIDDataset):
    """ImageNet VID dataset for video object detection."""

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, odd_working_mode="", *args, **kwargs):
        assert odd_working_mode and odd_working_mode in ["base", "agg"], "Please specify the working mode of ODDs"
        self.working_mode = 1 if odd_working_mode == "base" else 0
        self.cur_video_id = -1
        super().__init__(*args, **kwargs)

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        test_img_info = self.coco.load_imgs([1])[0]
        odd_label_flag = test_img_info.get('odd_label', None)
        if not odd_label_flag:
            for vid_id in self.vid_ids:
                img_ids = self.coco.get_img_ids_from_vid(vid_id)
                for img_id in img_ids:
                    info = self.coco.load_imgs([img_id])[0]
                    info['filename'] = info['file_name']
                    if self.test_mode:
                        assert not info['is_vid_train_frame'], \
                            'is_vid_train_frame must be False in testing'
                        self.img_ids.append(img_id)
                        data_infos.append(info)
                    elif info['is_vid_train_frame']:
                        self.img_ids.append(img_id)
                        data_infos.append(info)
            return data_infos
        else:
            for vid_id in self.vid_ids:
                img_ids = self.coco.get_img_ids_from_vid(vid_id)
                for img_id in img_ids:
                    info = self.coco.load_imgs([img_id])[0]
                    info['filename'] = info['file_name']
                    if self.test_mode:
                        assert not info['is_vid_train_frame'], \
                            'is_vid_train_frame must be False in testing'
                        if self.working_mode and info['odd_label']:
                            self.img_ids.append(img_id)
                            data_infos.append(info)
                        elif not self.working_mode and not info['odd_label']:
                            self.img_ids.append(img_id)
                            data_infos.append(info)
                    elif info['is_vid_train_frame']:
                        self.img_ids.append(img_id)
                        data_infos.append(info)
            return data_infos

    def ref_img_sampling(self,
                         img_info,
                         frame_range,
                         stride=1,
                         num_ref_imgs=1,
                         filter_key_img=True,
                         method='uniform',
                         return_key_img=True):
        """Sampling reference frames in the same video for key frame using global strategies.
        """
        assert isinstance(img_info, dict)
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')

        if 'test' in method and \
                (frame_range[1] - frame_range[0]) != num_ref_imgs:
            print_log(
                'Warning:'
                "frame_range[1] - frame_range[0] isn't equal to num_ref_imgs."
                'Set num_ref_imgs to frame_range[1] - frame_range[0].',
                logger=self.logger)
            self.ref_img_sampler[
                'num_ref_imgs'] = frame_range[1] - frame_range[0]

        if (not self.load_as_video) or img_info.get('frame_id', -1) < 0 \
                or (frame_range[0] == 0 and frame_range[1] == 0):
            ref_img_infos = []
            for i in range(num_ref_imgs):
                ref_img_infos.append(img_info.copy())
        else:
            vid_id, img_id, frame_id = img_info['video_id'], img_info[
                'id'], img_info['frame_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            left = max(0, frame_id + frame_range[0])
            right = min(frame_id + frame_range[1], len(img_ids) - 1)

            ref_img_ids = []
            if method == 'uniform':
                valid_ids = img_ids[left:right + 1]
                if filter_key_img and img_id in valid_ids:
                    valid_ids.remove(img_id)
                num_samples = min(num_ref_imgs, len(valid_ids))
                ref_img_ids.extend(random.sample(valid_ids, num_samples))
            elif method == 'bilateral_uniform':
                assert num_ref_imgs % 2 == 0, \
                    'only support load even number of ref_imgs.'
                for mode in ['left', 'right']:
                    if mode == 'left':
                        valid_ids = img_ids[left:frame_id + 1]
                    else:
                        valid_ids = img_ids[frame_id:right + 1]
                    if filter_key_img and img_id in valid_ids:
                        valid_ids.remove(img_id)
                    num_samples = min(num_ref_imgs // 2, len(valid_ids))
                    sampled_inds = random.sample(valid_ids, num_samples)
                    ref_img_ids.extend(sampled_inds)
            elif method == 'test_with_adaptive_stride':
                if frame_id == 0:
                    stride = float(len(img_ids) - 1) / (num_ref_imgs - 1)
                    for i in range(num_ref_imgs):
                        ref_id = round(i * stride)
                        ref_img_ids.append(img_ids[ref_id])
            elif method == 'test_with_fix_stride':
                if frame_id == 0:
                    for i in range(frame_range[0], 1):
                        ref_img_ids.append(img_ids[0])
                    for i in range(1, frame_range[1] + 1):
                        ref_id = min(round(i * stride), len(img_ids) - 1)
                        ref_img_ids.append(img_ids[ref_id])
                elif frame_id % stride == 0:
                    ref_id = min(
                        round(frame_id + frame_range[1] * stride),
                        len(img_ids) - 1)
                    ref_img_ids.append(img_ids[ref_id])
                img_info['num_left_ref_imgs'] = abs(frame_range[0]) \
                    if isinstance(frame_range, list) else frame_range
                img_info['frame_stride'] = stride
            elif method == 'test_global_normal':
                ref_ids = img_info['normal_global_ref_imgs']
                for ref_id in ref_ids:
                    ref_img_ids.append(img_ids[ref_id])
            elif method == 'test_global_odd':
                if vid_id != self.cur_video_id:
                    self.cur_video_id = vid_id
                    num_ref_global_images = min(num_ref_imgs, len(self.coco.videos[vid_id]['global_pool']))
                    for i in range(num_ref_global_images):
                        ref_img_ids.append(img_ids[self.coco.videos[vid_id]['global_pool'][i]])
            elif method == 'test_global_odd_v2':
                ref_ids = img_info['advice_global_ref_imgs']
                for ref_id in ref_ids:
                    ref_img_ids.append(img_ids[ref_id])
            elif method == 'test_local_odd':
                assert stride == 1, 'stride must be 1 in local odd sampling.'
                ref_ids = img_info['advice_ref_imgs']
                for ref_id in ref_ids:
                    ref_img_ids.append(img_ids[ref_id])
                img_info['num_left_ref_imgs'] = abs(frame_range[0]) \
                    if isinstance(frame_range, list) else frame_range
                img_info['frame_stride'] = stride
            elif method == 'test_mega_odd':
                ref_global_ids = img_info['advice_global_ref_imgs']
                ref_local_ids = img_info['advice_ref_imgs']
                for ref_local_id in ref_local_ids:
                    ref_img_ids.append(img_ids[ref_local_id])
                for ref_global_id in ref_global_ids:
                    ref_img_ids.append(img_ids[ref_global_id])
                img_info['num_left_ref_imgs'] = abs(frame_range[0]) \
                    if isinstance(frame_range, list) else frame_range
                img_info['frame_stride'] = stride
            else:
                raise NotImplementedError

            ref_img_infos = []
            for ref_img_id in ref_img_ids:
                ref_img_info = self.coco.load_imgs([ref_img_id])[0]
                ref_img_info['filename'] = ref_img_info['file_name']
                ref_img_infos.append(ref_img_info)
            ref_img_infos = sorted(ref_img_infos, key=lambda i: i['frame_id'])

        if return_key_img:
            return [img_info, *ref_img_infos]
        else:
            return ref_img_infos
