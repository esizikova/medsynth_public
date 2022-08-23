#!/bin/bash

cmd=( python3 -u train_joint_norm.py )   # create array with one element
cmd+=( --ymlpath=experiment/singleview2500/d2_singleview2500.yml )
cmd+=( --gpu=0 ) 
cmd+=( --dataroot=/LIDC-HDF5-256 ) 
cmd+=( --dataset=train )
cmd+=( --tag=d2_singleview2500 )
cmd+=( --data=LIDC256 )
cmd+=( --dataset_class=align_ct_xray_std )
cmd+=( --model_class=SingleViewCTGAN )
cmd+=( --datasetfile=data/train.txt )
cmd+=( --valid_datasetfile=data/test.txt )
cmd+=( --valid_dataset=test )
cmd+=( --save_path=save_models/singleView_CTGAN/LIDC256/d2_singleview2500_256/ )
# Run command
"${cmd[@]}"
