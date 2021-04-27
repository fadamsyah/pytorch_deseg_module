#!/bin/bash

filename=$1
img_folder=$2
use_cuda=$3

while read line; do
    `python inference.py --img_path $img_folder/$line --use_cuda $use_cuda`
done < $filename
