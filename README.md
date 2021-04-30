# pytorch_deseg_module

## Introduction
This repository is made as part of my internship at [Neurabot](https://neurabot.io). Shortly, the repository is used to detect objects in images in the field of medical imaging and pathology. Croped images then are segmented using an interactive segmentation technique. The object detection model needs data to learn whereas the segmentation model doesn't require any. The main references are:
- [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [shiyinzhang/Inside-Outside-Guidance](https://github.com/shiyinzhang/Inside-Outside-Guidance)

You can see modifications of the original repositories [here](zyolo_efficientdet/README.md) and [here](iog/README.md).

## Requirements
```
albumentations
numpy
opencv-contrib-python
pycocotools
torch
torchvision
```

## Installation
1. You need to install the requirements first.
2. Clone this repository `git clone --depth 1 https://github.com/fadamsyah/zyolo_efficientdet.git`
3. Install the cloned repository `pip install path/<zyolo_efficientdet>`

**Note:** Until now, you won't be able to install this repository because the `setup.py` has not been implemented. Please wait ...

## EfficientDet Training
For training, please refer to the [original repository](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). But, you should pay attention to the augmentation parameter on `projects/<project>.yml` ([examples](projects)).

Tips for training:
1. **Augmentation**. In my opinion, some of the most useful augmentation schemes in the [albumentation](https://github.com/albumentations-team/albumentations) library are:
   - Transpose
   - HorizontalFlip
   - VerticalFlip
   - ShiftScaleRotate
   - RandomCrop
   - MotionBlur
   - GaussianBlur
   - OneOf

    **Note**: You **MUST** visualize the augmented data to determine whether the augmentation process is relevant to your case. If you add improper augmentations to your training set, the result will be most likely **bad**.
2. In the begining of your training process, use `head_only=True` to train the head without updating the backbone and neck. Then, change to `head_only=False` after you see a saturation trend at training loss.
3. You may want to see [mnslarcher/kmeans-anchors-ratios](https://github.com/mnslarcher/kmeans-anchors-ratios) to determine optimal anchors ratios and scales.

## Inference
Coming soon ...