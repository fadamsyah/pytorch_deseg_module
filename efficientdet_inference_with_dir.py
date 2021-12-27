import argparse
import json
import cv2
import numpy as np
import os
import os.path as osp
import random
import yaml

from pytorch_deseg_module import DetectionModule
from pytorch_deseg_module.zyolo_efficientdet.utils.utils import boolean_string

def get_args():
    parser = argparse.ArgumentParser('Deteksi Sel Telur Fasciola')
    parser.add_argument('-c', '--det_compound_coef', type=int, default=2, help='coefficients of efficientdet')
    parser.add_argument('-p', '--project', type=str, default='sel_telur', help='object detection parameters')
    parser.add_argument('--image_dir', type=str, default=None, help='Images dir')
    parser.add_argument('--save_dir', type=str, default=None, help='Save dir')
    parser.add_argument('--use_cuda', type=boolean_string, default=False,
                        help='True kalau mau pake GPU yang ada CUDA nya. False kalau pakai CPU aja')
    parser.add_argument('--det_weights_path', type=str, default='weights/efficientdet-d2_80_4536.pth',
                        help='EfficientDet weights')
    parser.add_argument('--det_threshold', type=float, default=0.5,
                        help='persentase output min. untuk dianggap sebagai objek (0 - 1)')
    parser.add_argument('--det_iou_threshold', type=float, default=0.5,
                        help='persentase iou max. untuk menganggap 2 objek itu berbeda (0 - 1)')
    parser.add_argument('--text_font_scale', type=float, default=0.5, help='fontScale of category dan probability text')
    parser.add_argument('--text_font_thickness', type=int, default=1, help='font thickness of category and probability text')
    
    args = parser.parse_args()
    return args

def get_class_color(classes, seed=42):
    random.seed(seed)
    colors = {i: (random.randint(0, 255),
                  random.randint(0, 255),
                  random.randint(0, 255))
              for i in range(len(classes)+1)}
    
    return colors

def visualize(img_path, classes, results, fontScale, thickness, save_labeled_path=None, seed=42):
    img = cv2.imread(img_path)
    
    colors = get_class_color(classes, seed=seed)
    cat_to_idx = {category: i for i, category in enumerate(classes)}
        
    for obj in results:
        score = obj['score']
        
        category = obj['category']
        color = colors[cat_to_idx[category]]
        
        bb = obj['bbox']
        xmin, xmax, ymin, ymax = bb['xmin'], bb['xmax'], bb['ymin'], bb['ymax']
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        
        score = obj['score']
        cv2.putText(img, f'{score*100:.2f}: {category}', (xmin, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255,0,0),
                    thickness, cv2.LINE_AA)
        
    if save_labeled_path:
        cv2.imwrite(save_labeled_path, img)
    
    return img

if __name__ == "__main__":
    opt = get_args()
    params = yaml.safe_load(open(f'projects/{opt.project}.yml').read())
    classes = params['obj_list']
    
    # Create the EfficientNet model
    detector = DetectionModule(opt.det_compound_coef, params['obj_list'],
                               opt.det_weights_path, opt.use_cuda,
                               eval(params['anchors_ratios']),
                               eval(params['anchors_scales']))
    
    for image in os.listdir(opt.image_dir):
        # Image path
        image_path = osp.join(opt.image_dir, image)
        
        # Save path
        save_path = osp.join(opt.save_dir, image)
        
        # Detect objects
        det_outputs = detector(image_path, False,
                               opt.det_threshold,
                               opt.det_iou_threshold)
        
        results = det_outputs['analysis_results']
        
        # Visualize the result
        visualize(image_path, classes, results, opt.text_font_scale,
                  opt.text_font_thickness, save_path, seed=42)