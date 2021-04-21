# zyolo_efficientdet

## Introduction
This repository is a modifcation of [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). The main contribution of this repository is accommodating the use of the [Albumentations](https://github.com/albumentations-team/albumentations) library. The details will be explained in the following explanation.

## Modifications
### efficientdet/model.py
    # Add 2 lines of code to prevent blow up loss in the beginning of training
    # I forget where I found this solution. Please tell me if you know.

    self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False) # line 395
    self.header.pointwise_conv.conv.weight.data.fill_(0) # additional line
    self.header.pointwise_conv.conv.bias.data.fill_(-4.59) # additional line

### efficientdet/dataset.py
    # Add a new dataset class
    class CocoAlbumentationsDataset(CocoDataset):
        def __init__(...):
            ...

### utils/utils.py
    # Open an image from url
    def url_to_image(url):
        ...
    
    # Add the use_url argument on the preprocess method
    def preprocess(..., use_url=False):
        if not use_url:
            ori_imgs = [cv2.imread(img_path) for img_path in image_path]
        else:
            ori_imgs = [url_to_image(img_path) for img_path in image_path]
        ...

    # A class inheriting the COCOeval
    # which can be used to evaluate performance on every class
    class CustomCOCOeval(COCOeval):
        ...

### coco_eval.py
    # Add arguments of set_name and on_every_class
    ap.add_argument('--set_name', type=str, default='val_set', help='set name')
    ap.add_argument('--on_every_class', type=boolean_string, default=False, help='evaluate AP & AR for every class')

    # Modify the _eval method to accommodate an on_every_class evaluation
    def _eval(coco_gt, image_ids, pred_json_path, on_every_class):
      ...
      if on_every_class:
        for i in range(len(obj_list)):
            # Index starts from 1
            print('-------------------------------------')
            print(f'Evaluation on {obj_list[i]} class')
            coco_eval.params.catIds = [i+1]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
      ...

### train.py