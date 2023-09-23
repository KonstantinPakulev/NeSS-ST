#!/bin/bash

# ----------- MODEL SELECTION -------------------

models=(+models/shi_hardnet/default
        +models/sift_hardnet/default
        +models/superpoint_hardnet/default
        +models/r2d2_hardnet/default
        +models/keynet_hardnet/default
        +models/disk_hardnet/default
        +models/rekd_hardnet/default
        +models/shiness_hardnet/default)

# ------------- DATASET SELECTION ----------------

htune_dataset=+datasets/scannet/setup/ICCV2023/htune/rel_pose/e_pyopengv/num_2k
test_dataset=+datasets/scannet/setup/ICCV2023/test/rel_pose/e_pyopengv=num_2k

# -------------------------------------------------

#check=True
check=False

#for model in "${models[@]}"
#do
#  python3 source/pipeline/htune.py "$model=features_htune_test" \
#  $htune_dataset $check -f
#done

#for model in "${models[@]}"
#do
#  python3 source/pipeline/htune.py "$model=features_htune_test" \
#  $htune_dataset $check -lr -it
#done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
  $test_dataset $check -f
done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
  $test_dataset $check -t
done
