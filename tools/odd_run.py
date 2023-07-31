import argparse
import functools
import os
import os.path as osp
import time
from collections import defaultdict
import numpy as np
import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import set_random_seed
from mmdet.core import encode_mask_results

from odd.core import setup_multi_processes
from odd.datasets import build_dataset


determined_score_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
determined_score_anno_file_list = []

multi_process_flag = False

def global_advisor(anno_file, num_ref_image=14):
    cur_video_id = -1
    for cur_img_info in anno_file['images']:
        if not cur_img_info['odd_label']:
            ref_id_list = []
            if cur_img_info['video_id'] != cur_video_id:
                cur_video_id = cur_img_info['video_id']
                cur_global_pool = anno_file['videos'][cur_video_id - 1]['global_pool']
                num_ref_global_images = min(num_ref_image, len(cur_global_pool))
                for i in range(num_ref_global_images):
                    ref_id_list.append(cur_global_pool[i])
            # cur_img_info['advice_global_ref_imgs'] = ref_id_list
            cur_img_info['normal_global_ref_imgs'] = ref_id_list
        else:
            cur_img_info['normal_global_ref_imgs'] = []
    return anno_file


def global_advisor_normal(anno_file, num_ref_image=14):
    cur_video_id = -1
    video_len_list = []
    for cur_video_info in anno_file['videos']:
        video_len_list.append(len(cur_video_info['global_pool']))
    for cur_img_info in anno_file['images']:
        if not cur_img_info['odd_label']:
            ref_id_list = []
            if cur_img_info['video_id'] != cur_video_id:
                video_len = video_len_list[cur_img_info['video_id'] - 1]
                stride = float(video_len - 1) / (num_ref_image - 1)
                for i in range(num_ref_image):
                    ref_id = round(i * stride)
                    ref_id_list.append(ref_id)
            cur_img_info['normal_global_ref_imgs'] = ref_id_list
        else:
            cur_img_info['normal_global_ref_imgs'] = []
    return anno_file


def local_advisor(anno_file, stride=1, frame_range=[-15, 15]):
    video_len_list = []
    for cur_video_info in anno_file['videos']:
        video_len_list.append(len(cur_video_info['global_pool']))
    cur_video_id = -1
    local_max_right_id = -1
    for cur_img_info in anno_file['images']:
        if not cur_img_info['odd_label']:
            ref_id_list = []
            frame_id = cur_img_info['frame_id']
            video_len = video_len_list[cur_img_info['video_id'] - 1]
            max_left_ref_id = max(frame_id + round(frame_range[0] * stride), 0)
            if cur_img_info['video_id'] != cur_video_id or max_left_ref_id > local_max_right_id:
                cur_video_id = cur_img_info['video_id']
                for i in range(frame_range[0], 1):
                    ref_id = max(frame_id + round(i * stride), 0)
                    ref_id_list.append(ref_id)
                for i in range(1, frame_range[1] + 1):
                    ref_id = min(frame_id + round(i * stride), video_len - 1)
                    ref_id_list.append(ref_id)
                max_right_ref_id = frame_id + round(frame_range[1] * stride)
                local_max_right_id = max_right_ref_id
            else:
                if local_max_right_id < video_len - 1:
                    for i in range(frame_range[0], 1):
                        ref_id = max(frame_id + round(i * stride), 0)
                        if ref_id <= local_max_right_id:
                            continue
                        ref_id_list.append(ref_id)
                    for i in range(1, frame_range[1] + 1):
                        ref_id = min(frame_id + round(i * stride), video_len - 1)
                        if ref_id <= local_max_right_id:
                            continue
                        ref_id_list.append(ref_id)
                    max_right_ref_id = max(frame_id + round(frame_range[1] * stride), local_max_right_id)
                    local_max_right_id = max_right_ref_id
                else:
                    ref_id = video_len - 1
                    max_right_ref_id = frame_id + round(frame_range[1] * stride)
                    assert max_right_ref_id > local_max_right_id, 'max_right_ref_id must be larger than local_max_right_id'
                    bias = max_right_ref_id - local_max_right_id
                    for i in range(bias):
                        ref_id_list.append(ref_id)
                    local_max_right_id = max_right_ref_id
            cur_img_info['advice_ref_imgs'] = ref_id_list
        else:
            cur_img_info['advice_ref_imgs'] = []
    return anno_file


def split_class(ori_anno_path, out_result_path, cur_determined_score, out_anno_dir):
    new_file_path = os.path.join(out_anno_dir, 'anno_{}.json'.format(cur_determined_score))
    determined_score_anno_file_list.append(new_file_path)
    if os.path.exists(new_file_path):
        print("file exists, skip")
        return
    ori_anno = mmcv.load(ori_anno_path)
    out_result = mmcv.load(out_result_path)
    reg_score_list = []
    for cur_reg_score in out_result['iqa_reg']:
        reg_score_list.append(cur_reg_score.item())
    del out_result
    converted_anno_results = []
    for global_index in range(len(reg_score_list)):
        cur_anno = defaultdict()
        cur_anno.update(ori_anno['images'][global_index])
        cur_anno['score'] = reg_score_list[global_index]
        cur_anno['label'] = 1 if reg_score_list[global_index] > cur_determined_score else 0
        converted_anno_results.append(cur_anno)

    def cmp_score(a, b):
        if a['video_id'] < b['video_id']:
            return -1
        elif a['video_id'] > b['video_id']:
            return 1
        else:
            return -1 if a['score'] >= b['score'] else 1

    converted_anno_results.sort(key=functools.cmp_to_key(cmp_score))
    cur_global_pool = []
    for i in range(len(converted_anno_results)):
        cur_global_pool.append(converted_anno_results[i]['frame_id'])
        if i == len(converted_anno_results) - 1 or converted_anno_results[i]['video_id'] != \
                converted_anno_results[i + 1]['video_id']:
            vid_index = converted_anno_results[i]['video_id'] - 1
            assert ori_anno['videos'][vid_index]['id'] == converted_anno_results[i]['video_id']
            ori_anno['videos'][vid_index]['global_pool'] = cur_global_pool
            cur_global_pool = []
    for i, cur_image in enumerate(ori_anno['images']):
        cur_image['odds'] = reg_score_list[i]
        cur_image['odd_label'] = 1 if reg_score_list[i] > cur_determined_score else 0
    ori_anno = local_advisor(ori_anno)
    ori_anno = global_advisor_normal(ori_anno)
    mmcv.dump(ori_anno, new_file_path)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    fps=3,
                    show_score_thr=0.3,
                    fps_file_path=None,
                    out_result_path=None,
                    out_result_flag=None,
                    **kwargs):
    assert fps_file_path is not None, 'fps_file_path must be specified'
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    prev_img_meta = None
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data,
                           out_result_path=out_result_path, out_result_flag=out_result_flag)
        #  file_list.append(data['img_metas'][0]._data[0][0]['ori_filename'])
        batch_size = len(data['img'])
        # batch_size = data['img'][0].size(0)
        if show or out_dir:
            assert batch_size == 1, 'Only support batch_size=1 when testing.'
            img_tensor = data['img'][0]
            img_meta = data['img_metas'][0].data[0][0]
            img = tensor2imgs(img_tensor, **img_meta['img_norm_cfg'])[0]

            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if out_dir:
                out_file = osp.join(out_dir, img_meta['ori_filename'])
            else:
                out_file = None

            model.module.show_result(
                img_show,
                result,
                show=show,
                out_file=out_file,
                score_thr=show_score_thr)

            need_write_video = (
                    prev_img_meta is not None and img_meta['frame_id'] == 0
                    or i == len(dataset))
            if out_dir and need_write_video:
                prev_img_prefix, prev_img_name = prev_img_meta[
                    'ori_filename'].rsplit(os.sep, 1)
                prev_img_idx, prev_img_type = prev_img_name.split('.')
                prev_filename_tmpl = '{:0' + str(
                    len(prev_img_idx)) + 'd}.' + prev_img_type
                prev_img_dirs = f'{out_dir}/{prev_img_prefix}'
                prev_img_names = sorted(os.listdir(prev_img_dirs))
                prev_start_frame_id = int(prev_img_names[0].split('.')[0])
                prev_end_frame_id = int(prev_img_names[-1].split('.')[0])

                mmcv.frames2video(
                    prev_img_dirs,
                    f'{prev_img_dirs}/out_video.mp4',
                    fps=fps,
                    fourcc='mp4v',
                    filename_tmpl=prev_filename_tmpl,
                    start=prev_start_frame_id,
                    end=prev_end_frame_id,
                    show_progress=False)

            prev_img_meta = img_meta

        for key in result:
            if 'mask' in key:
                result[key] = encode_mask_results(result[key])

        for k, v in result.items():
            results[k].append(v)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def odd_run(fpsopt,
            work_dir,
            config,
            determined_score,
            cur_anno_file,
            checkpoint,
            out_result_path=None,
            out_result_flag=None,
            **kwargs):
    args = argparse.Namespace()
    args.fpsopt = fpsopt
    args.work_dir = work_dir
    args.config = config
    args.eval = None
    args.cfg_options = None
    args.eval_options = None
    args.fuse_conv_bn = False
    args.format_only = False
    args.gpu_collect = False
    args.local_rank = 0
    args.show = False
    args.show_dir = None
    args.show_score_thr = 0.3
    args.out = None
    args.tmpdir = None
    args.launcher = 'none'
    args.checkpoint = checkpoint
    assert args.fpsopt and args.fpsopt in ['base', 'agg'], "please input fps option"
    assert args.work_dir, "please input work_dir"
    fps_file_name = "fps_" + args.fpsopt + "_" + determined_score + ".json"
    out_file_name = "acc_out_" + args.fpsopt + "_" + determined_score + ".json"
    out_fps = os.path.join(args.work_dir, fps_file_name)
    out_val = os.path.join(args.work_dir, out_file_name)
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    cfg = Config.fromfile(args.config)
    if out_result_flag and out_result_path:
        cfg['data']['test']['ann_file'] = cur_anno_file
    from odd.apis import multi_gpu_test
    from odd.datasets import build_dataloader
    from odd.models import build_model
    # set multi-process settings
    global multi_process_flag
    if not multi_process_flag:
        setup_multi_processes(cfg)
        multi_process_flag = True
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('seed', None) is not None:
        set_random_seed(
            cfg.seed, deterministic=cfg.get('deterministic', False))
    cfg.gpu_ids = [0]
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        import datetime
        init_dist(args.launcher, timeout=datetime.timedelta(seconds=86400), **cfg.dist_params)
    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.log.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    if cfg.get('test_cfg', False):
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_model(cfg.model)

    # We need call `init_weights()` to load pretained weights in MOT task.
    model.init_weights()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    if not hasattr(model, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            show_score_thr=args.show_score_thr,
            fps_file_path=out_fps)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    if out_result_path and out_result_flag == False:
        mmcv.dump(outputs, out_result_path)
    mmcv.dump(outputs, out_val)
    return out_val, out_fps


def merge_result(work_dir, base_result_file, agg_result_file, determined_score, cur_anno_file):
    base_result = mmcv.load(base_result_file)
    agg_result = mmcv.load(agg_result_file)
    anno = mmcv.load(cur_anno_file)
    base_result = base_result['det_bboxes']
    agg_result = agg_result['det_bboxes']
    total_result = []
    base_model_index = 0
    agg_model_index = 0
    for i, cur_image in enumerate(anno['images']):
        if cur_image['odd_label'] == 1:
            total_result.append(base_result[base_model_index])
            base_model_index += 1
        else:
            total_result.append(agg_result[agg_model_index])
            agg_model_index += 1
    assert base_model_index == len(base_result)
    assert agg_model_index == len(agg_result)
    assert len(total_result) == len(base_result) + len(agg_result)
    final_result_dict = defaultdict()
    final_result_dict['det_bboxes'] = total_result
    output_file_name = os.path.join(work_dir, "acc_out_" + determined_score + ".json")
    os.remove(base_result_file)
    os.remove(agg_result_file)
    mmcv.dump(final_result_dict, output_file_name)
    return output_file_name


def odd_val(work_dir, config, result_file, eval, determined_score):
    args = argparse.Namespace()
    args.work_dir = work_dir
    args.config = config
    args.eval = eval
    args.cfg_options = None
    args.eval_options = None
    args.fuse_conv_bn = False
    args.format_only = False
    args.gpu_collect = False
    args.local_rank = 0
    args.show = False
    args.show_dir = None
    args.show_score_thr = 0.3
    args.out = None
    args.tmpdir = None
    args.launcher = 'none'
    args.checkpoint = None
    assert args.work_dir, "please input work_dir"
    assert args.eval, "please input eval type"
    outputs = mmcv.load(result_file)
    new_outputs = []
    for cur_out in outputs['det_bboxes']:
        cur_new_list = []
        for cur_box in cur_out:
            cur_new_list.append(np.array(cur_box)) if cur_box else cur_new_list.append(np.empty((0, 5)))
        new_outputs.append(cur_new_list)
    outputs['det_bboxes'] = new_outputs

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    from odd.datasets import build_dataloader
    from odd.models import build_model

    # set multi-process settings
    # setup_multi_processes(cfg)

    # set random seeds. Force setting fixed seed and deterministic=True in SOT
    # configs
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('seed', None) is not None:
        set_random_seed(
            cfg.seed, deterministic=cfg.get('deterministic', False))
    cfg.data.test.test_mode = True

    cfg.gpu_ids = [0]
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        import datetime
        init_dist(args.launcher, timeout=datetime.timedelta(seconds=86400), **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.log.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    if cfg.get('test_cfg', False):
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_model(cfg.model)
    # We need call `init_weights()` to load pretained weights in MOT task.
    model.init_weights()
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    if not hasattr(model, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    eval_hook_args = [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule', 'by_epoch'
    ]
    for key in eval_hook_args:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)
    metric_dict = dict(
        config=args.config, mode='test', epoch=cfg.total_epochs)
    metric_dict.update(metric)
    final_file_path = os.path.join(args.work_dir, 'odd_final_result' + determined_score + '.json')
    mmcv.dump(metric_dict, final_file_path)
    return final_file_path


def parse_args():
    parser = argparse.ArgumentParser(description='for odd auto level')
    parser.add_argument("--name")
    parser.add_argument("--siod-config", help="siod config file path")
    parser.add_argument("--siod-checkpoint", help="siod checkpoint file path")
    parser.add_argument("--oddvod-config", help="oddvod config file path")
    parser.add_argument("--oddvod-checkpoint", help="oddvod checkpoint file path")
    parser.add_argument("--vod-config", help="normal val config file path")
    args = parser.parse_args()
    return args

def evaluate_out_result_flag(out_result_path, ori_anno_path,out_anno_dir):
    if os.path.exists(out_result_path):
        for i in range(len(determined_score_list)):
            split_class(ori_anno_path, out_result_path, determined_score_list[i], out_anno_dir)
        return True
    return False

def odd_vod(agg_model_name,
            base_file,
            part1_checkpoint,
            part2_checkpoint,
            agg_file,
            val_file,
            global_anno_dir='data/ILSVRC/annotations/',
            base_work_dir='work_dirs/odds/'):
    global_work_dir = 'work_dirs/odds/{}/'.format(agg_model_name)
    ori_anno_path = os.path.join(global_anno_dir, 'imagenet_vid_val.json')
    out_result_path = os.path.join(base_work_dir, 'odd_score.pkl')
    out_result_flag = False
    out_anno_dir = os.path.join(global_anno_dir, 'odd_anno')
    prog_bar = mmcv.ProgressBar(len(determined_score_list))
    for i in range(len(determined_score_list)):
        # run base model
        print("run base model")
        if not out_result_flag:
            out_result_flag = evaluate_out_result_flag(out_result_path, ori_anno_path, out_anno_dir)
        cur_anno_file = 0 if not len(determined_score_anno_file_list) else determined_score_anno_file_list[i]
        base_output_file, base_fps = odd_run('base', global_work_dir, base_file, str(determined_score_list[i]),
                                             cur_anno_file, part1_checkpoint,
                                             out_result_path, out_result_flag)
        # run agg model
        print("run agg model")
        if not out_result_flag:
            out_result_flag = evaluate_out_result_flag(out_result_path, ori_anno_path, out_anno_dir)
        cur_anno_file = 0 if not len(determined_score_anno_file_list) else determined_score_anno_file_list[i]
        agg_output_file, agg_fps = odd_run('agg', global_work_dir, agg_file, str(determined_score_list[i]),
                                           determined_score_anno_file_list[i], part2_checkpoint)
        print("merge result")
        total_output_file = merge_result(global_work_dir, base_output_file, agg_output_file,
                                         str(determined_score_list[i]), cur_anno_file)
        print("evaluate model")
        final_file_path = odd_val(global_work_dir, val_file, total_output_file, ['bbox'],
                                  str(determined_score_list[i]))
        prog_bar.update()


def main():
    args = parse_args()
    # step 1: set env var
    assert args.name and args.siod_config and args.siod_checkpoint and args.oddvod_config and args.oddvod_checkpoint and args.vod_config
    agg_model_name = args.name
    base_file = args.siod_config
    part1_checkpoint = args.siod_checkpoint
    part2_checkpoint = args.oddvod_checkpoint
    agg_file = args.oddvod_config
    val_file = args.vod_config
    odd_vod(agg_model_name, base_file, part1_checkpoint, part2_checkpoint, agg_file, val_file)


if __name__ == "__main__":
    main()
