#!/bin/bash

checkpoint_path='/home/konstantin/personal/Summertime/runs/models/shiness/checkpoints/model_r_mAA=0.7706.pt'
image_path='/mnt/sda/datasets/IMCPT/british_museum/dense/images/99586182_328408105.jpg'
num_features=2048

python3 nessst.py $checkpoint_path $image_path $num_features
