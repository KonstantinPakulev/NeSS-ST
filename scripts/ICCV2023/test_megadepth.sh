#!/bin/bash

# ----------- MODEL SELECTION -------------------

models=(+models/shi_disk/default
        +models/sift_disk/default
        +models/superpoint_disk/default
        +models/r2d2_disk/default
        +models/keynet_disk/default
        +models/disk/default
        +models/rekd_disk/default
        +models/shiness_disk/default)

# ------------- DATASET SELECTION ----------------

test_dataset=+datasets/megadepth/setup/ICCV2023/test/rel_pose/f_pydegensac=num_2k

# -------------------------------------------------

import_dataset=imcpt

#check=True
check=False

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
    $test_dataset $check -f -ip $import_dataset
done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
    $test_dataset $check -t -ip $import_dataset
done
