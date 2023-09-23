#!/bin/bash

# ----------- MODEL SELECTION -------------------

models=(+models/shiss_disk/default
        +models/shirs_disk/default)

# ------------- ABLATION SELECTION ----------------

ablation=filtering

# ------------- DATASET SELECTION ----------------

htune_dataset=+datasets/imcpt/setup/ICCV2023/htune/rel_pose/f_pydegensac/num_2k
test_dataset=+datasets/imcpt/setup/ICCV2023/val/rel_pose/f_pydegensac=num_2k

# -------------------------------------------------

#check=True
check=False

for model in "${models[@]}"
do
  python3 source/pipeline/htune.py "$model=features_htune_test" \
     $htune_dataset $check -a $ablation -f
done

for model in "${models[@]}"
do
  python3 source/pipeline/htune.py "$model=features_htune_test" \
     $htune_dataset $check -a $ablation -lr -it
done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
     $test_dataset $check -a $ablation -f
done

for model in "${models[@]}"
do
  python3 source/pipeline/test.py "$model=features_htune_test" \
     $test_dataset $check -a $ablation -t
done
