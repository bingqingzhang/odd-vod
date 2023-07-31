import mmcv
import numpy as np
from collections import defaultdict
from mmdet.core import bbox_overlaps
import torch


def calculate_image_odd_score(cur_results, cur_anno_info, iou_thresh=0.5, sub_iou_thresh=0.3):
    tmp_pred = np.concatenate(cur_results, axis=0)
    pred_bbox = tmp_pred[:, 0:4]
    pred_score = tmp_pred[:, -1]
    pred_label = np.array([j for j in range(len(cur_results)) for k in range(len(cur_results[j]))])
    gt_bbox = np.array(cur_anno_info['bboxes'])
    gt_label = np.array(cur_anno_info['labels'])
    gt_ignore = np.zeros(len(gt_bbox))
    empty_weight = 0
    score = defaultdict(list)
    score_list = []
    match = defaultdict(list)
    match_list = []
    weighted_match_list = []
    pred_ignore = defaultdict(list)
    pred_ignore_list = []
    total_pos = 0
    for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
        pred_mask_l = pred_label == l
        pred_bbox_l = pred_bbox[pred_mask_l]
        pred_score_l = pred_score[pred_mask_l]
        # sort by score
        order = pred_score_l.argsort()[::-1]
        pred_bbox_l = pred_bbox_l[order]
        pred_score_l = pred_score_l[order]

        gt_mask_l = gt_label == l

        gt_bbox_l = gt_bbox[gt_mask_l]
        gt_ignore_l = gt_ignore[gt_mask_l]

        score[l].extend(pred_score_l)
        score_list.extend(pred_score_l)
        pred_ignore[l].extend((empty_weight,) * pred_bbox_l.shape[0])  # if needed, adjust this
        pred_ignore_list.extend((empty_weight,) * pred_bbox_l.shape[0])  # if needed, adjust this
        total_pos += gt_bbox_l.shape[0] - sum(gt_ignore_l)

        if len(pred_bbox_l) == 0:
            continue
        if len(gt_bbox_l) == 0:
            match[l].extend((0,) * pred_bbox_l.shape[0])
            match_list.extend((0,) * pred_bbox_l.shape[0])
            weighted_match_list.extend((0,) * pred_bbox_l.shape[0])
            #  pred_ignore[l].extend((empty_weight,) * pred_bbox_l.shape[0])
            continue
        pred_bbox_l = pred_bbox_l.copy()
        pred_bbox_l[:, 2:] += 1
        gt_bbox_l = gt_bbox_l.copy()
        gt_bbox_l[:, 2:] += 1
        bboxes1 = torch.FloatTensor(pred_bbox_l)
        bboxes2 = torch.FloatTensor(gt_bbox_l)
        iou = bbox_overlaps(bboxes1, bboxes2, is_aligned=False).numpy()
        # print(l, iou)
        num_obj, num_gt_obj = iou.shape
        for j in range(0, num_obj):
            arg_match = -1
            for k in range(0, num_gt_obj):
                if iou[j, k] < sub_iou_thresh:
                    continue
                if iou[j, k] < iou_thresh:
                    if arg_match == -1:
                        arg_match = 0
                else:
                    arg_match = 1
            if arg_match > 0:
                match[l].append(1)
                match_list.append(1)
                weighted_match_list.append(1)
            elif arg_match == 0:
                match[l].append(0)
                match_list.append(0)
                weighted_match_list.append(0.5)
            else:
                match[l].append(0)
                match_list.append(0)
                weighted_match_list.append(0)
    # now calculate weighted f1score
    match_array = np.array(match_list)
    weighted_match_array = np.array(weighted_match_list)
    score_array = np.array(score_list)
    pred_ignore_array = np.array(pred_ignore_list)
    # weighted_tps
    weighted_tps = np.sum(weighted_match_array * score_array * (1 - pred_ignore_array))
    # weighted_fps
    weighted_fps = np.sum((1 - match_array) * score_array * (1 - pred_ignore_array))
    if total_pos:
        weighted_precision = weighted_tps / (weighted_tps + weighted_fps + np.spacing(1))
    else:
        weighted_precision = (weighted_tps + 1) / (weighted_tps + 1 + weighted_fps + np.spacing(1))
    # weighted_precision = weighted_tps / (weighted_tps + weighted_fps + np.spacing(1))
    # weighted_recall = weighted_tps / total_pos
    weighted_recall = weighted_tps / total_pos if total_pos else 1.0
    # weighted_f1score
    weighted_f1score = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall + np.spacing(1))
    # print("results:", weighted_tps, weighted_fps, weighted_precision, weighted_recall, weighted_f1score)
    return weighted_f1score


def get_anno_info(image_info, total_annotations):
    image_id = image_info['id']
    l, r = 0, len(total_annotations) - 1
    while l < r:
        mid = (l + r) // 2
        if total_annotations[mid]['image_id'] >= image_id:
            r = mid
        else:
            l = mid + 1
    # assert total_annotations[l]['image_id'] == image_id, "image_id not found "+ str(total_annotations[l]['image_id'])
    if total_annotations[l]['image_id'] != image_id:
        # print("image_id not found "+ str(image_id))
        cur_anno_info = defaultdict(list)
        cur_anno_info['bboxes'] = []
        cur_anno_info['labels'] = []
        return cur_anno_info
    left_begin = l
    r = len(total_annotations) - 1
    while l < r:
        mid = (l + r + 1) // 2
        if total_annotations[mid]['image_id'] <= image_id:
            l = mid
        else:
            r = mid - 1
    assert total_annotations[l]['image_id'] == image_id, "image_id not found" + str(total_annotations[l]['image_id'],
                                                                                    image_id)
    right_begin = l
    cur_anno_info = defaultdict(list)
    for i in range(left_begin, right_begin + 1):
        x1, y1, w, h = total_annotations[i]['bbox']
        inter_w = max(0, min(x1 + w, image_info['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, image_info['height']) - max(y1, 0))
        if inter_w * inter_h == 0:
            continue
        if w < 1 or h < 1:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        cur_anno_info['labels'].append(total_annotations[i]['category_id'] - 1)
        cur_anno_info['bboxes'].append(bbox)
    return cur_anno_info


if __name__ == "__main__":
    # 加载数据
    anno_files = ['imagenet_det_30plus1cls.json', 'imagenet_vid_val.json', 'imagenet_vid_train.json ']
    anno_odd_files = ['imagenet_det_odd.json', 'imagenet_vid_val_odd.json', 'imagenet_vid_train_odd.json']
    result_files = ['results_det.pkl', 'results_val.pkl', 'results_train.pkl']
    odd_files = ['odd_det.json', 'odd_val.json', 'odd_train.json']
    for gi in range(0, 3):
        dir_base_path = "work_dirs/odds/"
        anno_base_path = "data/ILSVRC/annotations/"
        result_file_path = dir_base_path + result_files[gi]
        ori_anno_file_path = anno_base_path + anno_files[gi]
        f1_score_file_path = dir_base_path + odd_files[gi]
        result_dict = mmcv.load(result_file_path)
        anno_dict = mmcv.load(ori_anno_file_path)

        f1_score_list = []
        images_info_list = anno_dict['images']
        total_annotations = anno_dict['annotations']
        total_results = result_dict['det_bboxes']
        no_anno_image = []
        prog_bar = mmcv.ProgressBar(len(images_info_list))
        tot_index = 0
        for i in range(len(images_info_list)):
            prog_bar.update()
            if not gi and not images_info_list[i]['is_vid_train_frame']:
                continue
            cur_image = images_info_list[i]
            cur_anno_info = get_anno_info(cur_image, total_annotations)
            if not cur_anno_info['labels']:
                no_anno_image.append(cur_image['id'] - 1)
            cur_results = total_results[tot_index]
            cur_weighted_f1score = calculate_image_odd_score(cur_results, cur_anno_info)
            f1_score_list.append(cur_weighted_f1score)
            if i in [0, 2]:
                images_info_list[i]['is_vid_train_frame'] = True
                images_info_list[i]['quality'] = cur_weighted_f1score
            else:
                images_info_list[i]['quality'] = cur_weighted_f1score
            tot_index += 1
        f1_score_dict = defaultdict()
        f1_score_dict['f1_score'] = f1_score_list
        f1_score_dict['no_anno_image'] = no_anno_image
        mmcv.dump(f1_score_dict, f1_score_file_path)
        mmcv.dump(anno_dict, anno_odd_files[gi])