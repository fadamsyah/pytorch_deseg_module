# zyolo_efficientdet

## Introduction
This repository is a modifcation of [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). The main contribution of this repository is accommodating the use of the [Albumentations](https://github.com/albumentations-team/albumentations) library. The details are explained [here](zyolo_efficientdet/README.md).

## Note
The details of modifications from the original repository will be published soon ...

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