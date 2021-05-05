import argparse
import json
import cv2
import numpy as np
import os
import yaml

from pytorch_deseg_module import DetectionModule
from pytorch_deseg_module.zyolo_efficientdet.utils.utils import boolean_string
from pytorch_deseg_module import IoGNetwork

def get_args():
    parser = argparse.ArgumentParser('Deteksi Sel Telur Fasciola')
    parser.add_argument('--project', type=str, default='sel_telur', help='object detection parameters')
    parser.add_argument('--img_path', type=str, default='test/sample.jpg',
                        help='path image nya')
    parser.add_argument('--use_url', type=boolean_string, default=False,
                        help='True apabila menggunakan url untuk load gambarnya. False kalau load gambar dari local')
    parser.add_argument('--use_cuda', type=boolean_string, default=False,
                        help='True kalau mau pake GPU yang ada CUDA nya. False kalau pakai CPU aja')
    parser.add_argument('--det_compound_coef', type=int, default=2, help='coefficients of efficientdet')
    parser.add_argument('--det_weights_path', type=str, default='weights/efficientdet-d2_80_4536.pth',
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
    params = yaml.safe_load(open(f'projects/{opt.project}.yml').read())
    
    # Create the EfficientNet model
    detector = DetectionModule(opt.det_compound_coef, params['obj_list'],
                               opt.det_weights_path, opt.use_cuda,
                               eval(params['anchors_ratios']),
                               eval(params['anchors_scales']))
    
    # Create the IoG segmentation model
    iog = IoGNetwork(pretrain_path=opt.iog_weights_path,
                     interpolation=cv2.INTER_CUBIC,
                     use_cuda=opt.use_cuda,
                     threshold=0.9)
    
    # Detect objects
    det_outputs = detector(opt.img_path, opt.use_url,
                           opt.det_threshold, opt.det_iou_threshold)
    
    # Segment objects
    iog_outputs = iog(opt.img_path, det_outputs)
    iog_outputs = (iog_outputs * 255).astype(np.uint8)
    iog_outputs = cv2.cvtColor(iog_outputs, cv2.COLOR_GRAY2RGB)
    
    # Get the image name
    base_img_name = os.path.basename(opt.img_path)
    base_img_name, _ = os.path.splitext(base_img_name)
    
    # Save the segmentation output
    cv2.imwrite(f'test/segmentation/{base_img_name}.jpg', iog_outputs)
    
    # Save the bounding-boxes output
    json_object = json.dumps(det_outputs, indent = 4)
    with open(f'test/bbs/{base_img_name}.json', "w") as outfile: 
        outfile.write(json_object)
    
    # img = cv2.imread(opt.img_path)
    # bbox = det_outputs['analysis_results'][0]['bbox']
    # concatenated = np.concatenate(
    #                 (img[bbox['ymin']:bbox['ymax'], bbox["xmin"]:bbox["xmax"]],
    #                  iog_outputs[bbox['ymin']:bbox['ymax'], bbox["xmin"]:bbox["xmax"]]),
    #                 axis=1)
    
    # cv2.imwrite(f'test/result_1.jpg', concatenated)
    # cv2.imwrite(f'test/result_2.jpg', iog_outputs)