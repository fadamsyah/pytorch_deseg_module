import argparse
import json
import cv2
import numpy as np
import os

from zyolo_efficientdet import DetectionModule
from iog import IoGNetwork

def get_args():
    parser = argparse.ArgumentParser('Deteksi Sel Telur Fasciola')
    parser.add_argument('--img_path', type=str, default='datasets/malaria/train/d9345456-b76d-46cc-b81e-46d4a0a7b652.png',
                        help='path image nya')
    parser.add_argument('--use_url', type=boolean_string, default=False,
                        help='True apabila menggunakan url untuk load gambarnya. False kalau load gambar dari local')
    parser.add_argument('--use_cuda', type=boolean_string, default=False,
                        help='True kalau mau pake GPU yang ada CUDA nya. False kalau pakai CPU aja')
    parser.add_argument('--det_compound_coef', type=int, default=2, help='coefficients of efficientdet')
    parser.add_argument('--det_weights_path', type=str, default='weights/efficientdet-d2_15_1712.pth',
                        help='EfficientDet weights')
    parser.add_argument('--det_threshold', type=float, default=0.5,
                        help='persentase output min. untuk dianggap sebagai objek (0 - 1)')
    parser.add_argument('--det_iou_threshold', type=float, default=0.5,
                        help='persentase iou max. untuk menganggap 2 objek itu berbeda (0 - 1)')
    parser.add_argument('--iog_weights_path', type=str, default='models/IOG_PASCAL_SBD.pth',
                        help='IoG weights')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = get_args()
    
    # Create the EfficientNet model
    detector = DetectionModule(opt.det_compound_coef, obj_list, opt.det_weights_path,
                               opt.use_cuda, anchors_ratios, anchors_scales)
    
    # Create the IoG segmentation model
    iog = IoGNetwork(pretrain_path=opt.iog_weights_path, use_cuda=opt.use_cuda)
    
    # Detect objects
    det_outputs = detector(opt.img_path, opt.use_url,
                           opt.det_threshold, opt.det_iou_threshold)
    
    # Segment objects
    iog_outputs = iog(opt.img_path, det_outputs)