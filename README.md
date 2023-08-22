# Object Detection Difficulty: Suppressing Over-aggregation for Faster and Better Video Object Detection



> Current video object detection (VOD) models often encounter issues with over-aggregation due to redundant aggregation strategies, which perform feature aggregation on every frame. This results in suboptimal performance and increased computational complexity. In this work, we propose an image-level Object Detection Difficulty (ODD) metric to quantify the difficulty of detecting objects in a given image. The derived ODD scores can be used in the VOD process to mitigate over-aggregation. Specifically, we train an ODD predictor as an auxiliary head of a still-image object detector to compute the ODD score for each image based on the discrepancies between detection results and ground-truth bounding boxes. The ODD score enhances the VOD system in two ways: 1) it enables the VOD system to select superior global reference frames, thereby improving overall accuracy; and 2) it serves as an indicator in the newly designed ODD Scheduler to eliminate the aggregation of frames that are easy to detect, thus accelerating the VOD process. Comprehensive experiments demonstrate that, when utilized for selecting global reference frames, ODD-VOD consistently enhances the accuracy of Global-frame-based VOD models. When employed for acceleration, ODD-VOD consistently improves the frames per second (FPS) by an average of $73.3\%$ across 8 different VOD models without sacrificing accuracy. When combined, ODD-VOD attains state-of-the-art performance when competing with many VOD methods in both accuracy and speed. Our work represents a significant advancement towards making VOD more practical for real-world applications. 

## Installation

```shell
conda create -n odd python=3.8 -y
conda activate odd
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmdet
cd odd
pip install -r requirements/build.txt
pip install -v -e .  
```

## Dataset Preparation

`Step1` Download ImagenetVID and ImagenetDET from [ILSVRC](http://image-net.org/challenges/LSVRC/2017/).

`Step2` Data Structure

```
odd
├── data
│   └── ILSVRC
│       ├── annotations
│       ├── Annotations
│       ├── Data 
│       └── Lists 
...
```
* The `Lists` under `ILSVRC` contains the txt files from [here](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

`Step3` Convert Annotations

```shell
# ImageNet DET
python ./tools/convert_datasets/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations
```

`NOTE`
There are 3 JSON files in `data/ILSVRC/annotations`:

`imagenet_det_30plus1cls.json`: JSON file containing the annotations information of the training set in ImageNet DET dataset. The `30` in `30plus1cls` denotes the overlapped 30 categories in ImageNet VID dataset, and the `1cls` means we take the other 170 categories in ImageNet DET dataset as a category, named as `other_categeries`.

`imagenet_vid_train.json`: JSON file containing the annotations information of the training set in ImageNet VID dataset.

`imagenet_vid_val.json`: JSON file containing the annotations information of the validation set in ImageNet VID dataset.

## Training ODD-VOD from Scratch

### 1. Train a still-image detector (SIOD) and video object detector.

* Train Faster R-CNN (SIOD)

```shell
python tools/train.py configs/vid/faster_rcnn/faster_rcnn_r50_imagenetvid.py
```

* Train SELSA (VOD)

```shell
python tools/train.py configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py
```

### 2. Train the ODD Predictor with SIOD

* Generate Detection Results

```shell
python tools/test.py configs/vid/faster_rcnn/faster_rcnn_r50_imagenetvid.py --checkpoint checkpoints/faster_rcnn_r50.pth --out work_dirs/odd/results_val.pkl
```
```shell
python tools/test.py configs/vid/faster_rcnn/faster_rcnn_r50_imagenetvid_gener_det_only.py --checkpoint checkpoints/faster_rcnn_r50.pth --out work_dirs/odd/results_det.pkl
```
```shell
python tools/test.py configs/vid/faster_rcnn/faster_rcnn_r50_imagenetvid_gener_train_only.py --checkpoint checkpoints/faster_rcnn_r50.pth --out work_dirs/odd/results_train.pkl
```

* Calculate ODD Ground Truth Score

```shell
python tools/calculate_odd_score.py
```

* Train ODD Predictor on Faster R-CNN

```shell
python tools/train.py configs/vid/odd/odd_faster_rcnn_r50_imagenet_vid.py --no-validate
```

### 3. Run ODD-VOD

```
python tools/odd_run.py --name selsa_r50 --siod-config configs/vid/odd/odds_part1_fasterrcnn_r50_imagenetvid.py --siod-checkpoint checkpoints/odd_faster_rcnn_r50.pth --oddvod-config configs/vid/odd/odds_part2_selsa_fasterrcnn_r50_imagenetvid.py --oddvod-checkpoint checkpoints/selsa_r50.pth --vod-config configs/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py
```
There are 6 parameters:

* `--name`: the working dir name
* `--siod-config`: the config file for SIOD
* `--siod-checkpoint` the model weights of SIOD
* `--oddvod-config` the config file for VOD
* `--oddvod-checkpoint` the model weights of VOD
* `--vod-config` the normal vod config file, which is used to evaluate the final results

