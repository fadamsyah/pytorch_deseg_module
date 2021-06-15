import argparse
import os
import yaml
import math
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from pytorch_deseg_module.zyolo_efficientdet.efficientdet.dataset import CocoAlbumentationsDataset
from pytorch_deseg_module.zyolo_efficientdet.utils.utils import boolean_string
from pytorch_deseg_module import IoGNetwork

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
    
def get_args():
    parser = argparse.ArgumentParser('Object Detection Dataset Visualization')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--set_name', type=str, default='train', help='the set name')
    parser.add_argument('--idx', type=int, default=0, help='choose a sample from dataset using this index')
    parser.add_argument('--image_name', type=str, default=None, help='file name if want to save the visualization')
    parser.add_argument('--use_cuda', type=boolean_string, default=False,
                        help='True kalau mau pake GPU yang ada CUDA nya. False kalau pakai CPU aja')
    parser.add_argument('--iog_weights_path', type=str, default='models/IOG_PASCAL_SBD.pth',
                        help='IoG weights')
    parser.add_argument('--iog_batch_size', type=int, default=4, help='batch size for IoG')
    parser.add_argument('--iog_num_workers', type=int, default=4, help='the number of workers for IoG dataloader')
    
    args = parser.parse_args()
    return args

def read_json(path):
    f = open(path,)
    data = json.load(f)
    f.close()

    return data

if __name__ == '__main__':
    opt = get_args()
    params = Params(f'projects/{opt.project}.yml')
    
    # Read the annotation file
    annotations = read_json(
        os.path.join(opt.data_path, params.project_name, 'annotations', f'instances_{opt.set_name}.json')
    )
    
    # Get a sample
    image = annotations['images'][opt.idx]
    image_id = image['id']
    image_path = os.path.join(opt.data_path, params.project_name, opt.set_name, image['file_name'])
    
    input_annotations = {'analysis_results': []}
    for annotation in annotations['annotations']:
        if annotation['image_id'] != image_id:
            continue
        x1, y1, x2, y2 = annotation['bbox']
        entity = {
            "bbox": {
                "xmax": int(x2 + x1 + 0.5),
                "xmin": int(x1 + 0.5),
                "ymax": int(y2 + y1 + 0.5),
                "ymin": int(y1 + 0.5)
            },
            "category": params.obj_list[annotation['category_id'] - 1],
            "score": None
        }
        input_annotations['analysis_results'].append(entity)

    # Create the IoG segmentation model
    iog = IoGNetwork(pretrain_path=opt.iog_weights_path,
                     interpolation=cv2.INTER_CUBIC,
                     use_cuda=opt.use_cuda,
                     threshold=0.9,
                     batch_size=opt.iog_batch_size,
                     num_workers=opt.iog_num_workers)
    
    # Segment objects
    iog_outputs = iog(image_path, input_annotations)
    iog_outputs = (iog_outputs * 255).astype(np.uint8)
    iog_outputs = cv2.cvtColor(iog_outputs, cv2.COLOR_GRAY2RGB)
    
    # Save the segmentation output
    if opt.image_name is not None:
        cv2.imwrite(f'test/output_{opt.image_name}', iog_outputs)
        cv2.imwrite(f'test/original_{opt.image_name}', cv2.imread(image_path))