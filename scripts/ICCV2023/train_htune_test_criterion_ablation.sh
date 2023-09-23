#!/bin/bash

# ----------- MODEL SELECTION -------------------

models=(+models/logness_disk/default
        +models/dohness_disk/default
        +models/shiness_disk/default)
criteria=(+criteria=ss_loss/log
          +criteria=ss_loss/doh
          +criteria=ss_loss/shi)

#models=(+models/shiness_disk/default)
#criteria=(+criteria=ss_loss/shi)

# ------------- ABLATION SELECTION ----------------

ablation=filtering
#ablation=homography

# ------------- DATASET SELECTION ----------------

train_dataset=+datasets/megadepth/setup/ICCV2023=train
val_dataset=+datasets/imcpt/setup/ICCV2023/train_val/rel_pose/f_pydegensac=num_2k
htune_dataset=+datasets/imcpt/setup/ICCV2023/htune/rel_pose/f_pydegensac/num_2k
test_dataset=+datasets/imcpt/setup/ICCV2023/val/rel_pose/f_pydegensac=num_2k

# -------------------------------------------------

optimizer=+optimizer=default

#check=True
check=False

for i in "${!models[@]}"
do
  python3 source/pipeline/train.py "${models[i]}=train" \
   $train_dataset "${criteria[i]}" $optimizer $val_dataset $check -a $ablation
done

for i in "${!models[@]}"
do
python3 source/pipeline/htune.py "${models[i]}=features_htune_test" \
   $htune_dataset $check -a $ablation -c "${criteria[i]}"
done

for i in "${!models[@]}"
do
  python3 source/pipeline/test.py "${models[i]}=features_htune_test" \
     $test_dataset $check -a $ablation -c "${criteria[i]}"
done
