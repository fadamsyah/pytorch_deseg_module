# PyTorch: Detection and Segmentation Module

## Introduction
This repository is made as part of my internship at [Neurabot](https://neurabot.io). Shortly, the repository is used to detect objects in images in the field of medical imaging and pathology. Croped images then are segmented using an interactive segmentation technique. The object detection model needs data to learn whereas the segmentation model doesn't require any. The main references are:
- [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [shiyinzhang/Inside-Outside-Guidance](https://github.com/shiyinzhang/Inside-Outside-Guidance)

You can see modifications of the original repositories [here](pytorch_deseg_module/zyolo_efficientdet/README.md) and [here](pytorch_deseg_module/iog/README.md).

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
1. You need to install the **requirements** first.
2. Clone this repository `git clone --depth 1 https://github.com/fadamsyah/pytorch_deseg_module.git`
3. Install the cloned repository `pip install pytorch_deseg_module`

**Example**:
```bash
foo@bar:~$ git clone --depth 1 https://github.com/fadamsyah/pytorch_deseg_module.git
foo@bar:~$ pip install pytorch_deseg_module
```


## EfficientDet Training
The pretrained weights are available on the [original repository](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). For training, please refer to the original repository. But, you should pay attention to the augmentation parameter on `projects/<project>.yml` ([examples](projects)). You are not encouraged to train an EfficientDet model from scratch unless you have a lot of computing resources and data. From my experience, I only need to do the transfer learning technique to produce a considerably well and robust model.

This is how you should prepare your folder for training a EfficientDet model:

```
# You are highly recommended to structure your project folder as follows
project/
    datasets/
        {your_project_name}/
            annotations/
                - instances_train.json
                - instances_val.json
                - instances_test.json
            train/
                - *.jpg
            val/
                - *.jpg
            test/
                - *.jpg
    logs/
        # The training history will be written as the tensorboard format
        {your_project_name}/
            tensorboard/
    projects/
        # Put your project description here
        - {your_project_name}.yml
    weights/
        # Parameters of the trained model will be automatically saved
        - *.pth
    efficientdet_train.py
```

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

## CLI Examples
```bash
# Visualize a sample of dataset
python efficientdet_dataset_viz.py -p ki67 --set_name val --transform False --resize False --idx 0

# A sample training command
python efficientdet_train.py -p ki67 -c 0 --head_only True --lr 5e-4 --weight_decay 1e-5 --batch_size 16 --load_weights weights.pth --num_epochs 20

# Evaluate an EffDet model
python efficientdet_coco_eval.py -p ki67 -c 0 -w weights.pth --set_name val --on_every_class True --cuda True

# A sample inference
python inference.py --project ki67 --img_path image.jpg --use_cuda True --det_compound_coef 0 --det_weights_path effdet_weights.pth --iog_weights_path iog_weights.pth
```

## TODO
- [X] Add a code to visualize object detection dataset.
- [X] Add a code to visualize iog segmentation from dataset.
- [ ] Save the last parameters & the best parameters when training `efficientdet_train.py`.
- [ ] Generalize the [IoGNetwork](pytorch_deseg_module/iog/iog.py) for multi-class segmentation.
- [X] Use the PyTorch dataloader on [IoGNetwork](pytorch_deseg_module/iog/iog.py) to specify the batch_size when inferencing.