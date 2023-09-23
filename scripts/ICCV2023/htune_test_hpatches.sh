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

#models=(+models/harris_disk/default
#        +models/doh_disk/default
#        +models/log_disk/default
#        +models/dohness_disk/default
#        +models/logness_disk/default)

#models=(+models/shiss_disk/default
#        +models/shirs_disk/default
#        +models/shiners_disk/default)

# ------------- DATASET SELECTION ----------------

test_mma_dataset=+datasets/hpatches/setup/ICCV2023/test/classical/mma=num_2k
test_rep_dataset=+datasets/hpatches/setup/ICCV2023/test/classical/rep=num_2k

htune_dataset=+datasets/hpatches/setup/ICCV2023/htune/rel_pose/h_opencv/num_2k
test_dataset=+datasets/hpatches/setup/ICCV2023/test/rel_pose/h_opencv=num_2k

# -------------------------------------------------

#check=True
check=False

# ------------------ CLASSICAL METRICS ------------------

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
  $test_mma_dataset $check -f
done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
  $test_mma_dataset $check -t
done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
  $test_rep_dataset $check -t
done

# ------------------ HOMOGRAPHY ESTIMATION ------------------

#for model in "${models[@]}"
#do
#  python3 source/pipeline/htune.py "$model=features_htune_test" \
#  $htune_dataset $check -f
#done
#
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
