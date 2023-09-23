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

htune_dataset=+datasets/imcpt/setup/ICCV2023/htune/rel_pose/f_pydegensac/num_2k
test_dataset=+datasets/imcpt/setup/ICCV2023/test/rel_pose/f_pydegensac=num_2k

# -------------------------------------------------

#check=True
check=False

num_features=(128 512)

for model in "${models[@]}"
do
  python3 source/pipeline/htune.py "$model=features_htune_test" \
                $htune_dataset $check -lr -it -nf "${num_features[@]}"
done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
                $test_dataset $check -t -nf "${num_features[@]}"
done
