import argparse
import os
import yaml
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

from pytorch_deseg_module.zyolo_efficientdet.efficientdet.dataset import CocoAlbumentationsDataset
from pytorch_deseg_module.zyolo_efficientdet.utils.utils import boolean_string

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def get_args():
    parser = argparse.ArgumentParser('Object Detection Dataset Visualization')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--set_name', type=str, default='train', help='the set name')
    parser.add_argument('--resize', type=boolean_string, default=False, help='whether to resize the image or not')    
    parser.add_argument('--transform', type=boolean_string, default=False,
                        help='whether to augment dataset as in the corresponding project file')
    parser.add_argument('--idx', type=int, default=0, help='choose a sample from dataset using this index')
    args = parser.parse_args()
    return args

TEXT_COLOR = (255, 255, 255) # White
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = np.array(image).copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    cv2.imshow('dataset visualization', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    opt = get_args()
    params = Params(f'projects/{opt.project}.yml')
    
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    
    # Create the dataset
    if opt.transform:
        dataset = CocoAlbumentationsDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                                            set=eval(f'params.{opt.set_name}_set'),
                                            transform=A.Compose([eval(params.augmentation[i])
                                                                 for i in range(len(params.augmentation))],
                                                                bbox_params=A.BboxParams(format='coco',
                                                                                         label_fields=['category_ids'],
                                                                                         min_visibility=0.2),),
                                            img_size=input_sizes[opt.compound_coef], resize=opt.resize)
    else:
        dataset = CocoAlbumentationsDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                                            set=eval(f'params.{opt.set_name}_set'), transform=None,
                                            img_size=input_sizes[opt.compound_coef], resize=opt.resize)
        
    # Get a sample
    sample = dataset[opt.idx]

    # Visualize the dataset
    visualize(
        sample['img'],
        sample['annot'],
        [math.ceil(annot[-1]) for annot in sample['annot']],
        {i: category for i, category in enumerate(params.obj_list)},
    )