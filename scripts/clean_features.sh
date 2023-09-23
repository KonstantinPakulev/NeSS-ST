#!/bin/bash

dataset="scannet"
#dataset="megadepth"
#dataset="imcpt"
#dataset="hpatches"

#evaluation_type="htune"
evaluation_type="test"

find runs/$evaluation_type/$dataset -type f -iname *.h5py -delete