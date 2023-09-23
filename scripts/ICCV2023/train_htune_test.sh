#!/bin/bash

# ----------- MODEL SELECTION -------------------

models=(+models/shiness_disk/default
        +models/shiners_disk/default)

criteria=(+criteria=ss_loss/shi
          +criteria=rs_loss/shi)

# ------------- DATASET SELECTION ----------------

train_dataset=+datasets/megadepth/setup/ICCV2023=train
val_dataset=+datasets/imcpt/setup/ICCV2023/train_val/rel_pose/f_pydegensac=num_2k
htune_dataset=+datasets/imcpt/setup/ICCV2023/htune/rel_pose/f_pydegensac/num_2k
test_dataset=+datasets/imcpt/setup/ICCV2023/test/rel_pose/f_pydegensac=num_2k

# -------------------------------------------------

optimizer=+optimizer=default

#check=True
check=False

for i in "${!models[@]}"
do
  python3 source/pipeline/train.py "${models[i]}=train" \
    $train_dataset "${criteria[i]}" $optimizer $val_dataset $check
done

for i in "${!models[@]}"
do
  python3 source/pipeline/htune.py "${models[i]}=features_htune_test" \
    $htune_dataset $check
done

for i in "${!models[@]}"
do
  python3 source/pipeline/test.py "${models[i]}=features_htune_test" \
    $test_dataset $check
done
