# NeSS-ST: Detecting Good and Stable Keypoints with a Neural Stability Score and the Shi-Tomasi Detector

## <p align="center">[Arxiv pre-print](https://arxiv.org/abs/2307.01069) | [ICCV 2023 camera-ready](https://openaccess.thecvf.com/content/ICCV2023/papers/Pakulev_NeSS-ST_Detecting_Good_and_Stable_Keypoints_with_a_Neural_Stability_ICCV_2023_paper.pdf) | [Supplementary material](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Pakulev_NeSS-ST_Detecting_Good_ICCV_2023_supplemental.pdf) | [Video](https://youtu.be/qnP3agH3FAI)</p> 

## Table of contents

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Scripts](#scripts)
5. [Results](#results)

## Installation

To ease reproducibility of our results, we provide all dependencies in a Docker container. Launching of scripts and running Jupyter notebooks is done from the container.

Firstly, clone the respository:
```shell
git clone https://github.com/KonstantinPakulev/NeSS-ST
```

Then build a Docker image:
```shell
cd NeSS-ST
docker build . -t nessst
```

Next, run the container in the detached mode and enter it:
```shell
docker run -p 8888:8888 -p 6006:6006 --hostname $HOSTNAME -v $PATHTONESSST:/home/konstantin/personal/Summertime -v $PATHTODATASETS:/mnt/sda --shm-size=16384m --gpus all -dit nessst
docker exec -it $CONTAINERID bash
```

## Datasets

The project expects all datasets to be stored in `$PATHTODATASETS/datasets` folder. We describe steps required to prepare each of the datasets to be used by our pipeline. If you don't follow the naming conventions, you will need to modify configuration files in [config](config) on your own.

### HPatches
Downdload the sequences, unpack and place them in `$PATHTODATASETS/datasets/HPatches`:

```shell
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar xvzf hpatches-sequences-release.tar.gz
mv hpatches-sequences-release $PATHTODATASETS/datasets/HPatches
```

You can download the data splits that we use from [here](https://drive.google.com/file/d/1NMZGz15Px0g_Rw5EJj5LM_O2L_7g-PBo/view?usp=sharing) and put them in the `HPatches` directory, or you can generate the splits yourself by using the code from a [Jupyter notebook](notebooks/hpatches/dataset.ipynb).

### IMC-PT
Download validation and test sequences, unpack and place them in `$PATHTODATASETS/datasets/IMCPT`:
```shell
wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/ValidationData/imw2020-valid.tar.gz
tar xvzf imw2020-valid.tar.gz
mv imw2020-valid/* $PATHTODATASETS/datasets/IMCPT

wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/imw2020-test.tar.gz
tar xvzf imw2020-test.tar.gz
mv imw2020-test/* $PATHTODATASETS/datasets/IMCPT
```

You can download the data splits that we use from [here](https://drive.google.com/file/d/1NMZGz15Px0g_Rw5EJj5LM_O2L_7g-PBo/view?usp=sharing) and put them in the `IMCPT` directory, or you can generate the splits yourself by using the code from a [Jupyter notebook](notebooks/imcpt/dataset.ipynb).

### MegaDepth

Download the dataset, unpack and place it in `$PATHTODATASETS/datasets/MegaDepth`:
```shell
wget http://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz
tar xvzf MegaDepth_v1.tar.xz
mv MegaDepth_v1 $PATHTODATASETS/datasets/MegaDepth/MegaDepth_v1

wget http://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz
tar xvzf MegaDepth_SfM_v1.tar.xz
mv MegaDepth_SfM_v1 $PATHTODATASETS/datasets/MegaDepth/MegaDepth_SfM_v1
```

We use the code from the [D2-Net repository](https://github.com/mihaidusmanu/d2-net) for processing MegaDepth:

```shell
cd source/datasets/megadepth/preprocessing/base
sh create_reconstruction.sh
sh preprocess_undistorted_megadepth.sh
```

You can download the data splits that we use from [here](https://drive.google.com/file/d/1NMZGz15Px0g_Rw5EJj5LM_O2L_7g-PBo/view?usp=sharing) and put them in the `MegaDepth/SceneInfo` directory, or you can generate the splits yourself by using the code from a [Jupyter notebook](notebooks/megadepth/dataset.ipynb). If you choose the latter option, you will need to download the MegaDepth splits file used by DISK and put it into the `$PATHTODATASETS/datasets/MegaDepth/SceneInfo` directory:

```shell
wget https://datasets.epfl.ch/disk-data/megadepth/dataset.json
mv dataset.json $PATHTODATASETS/datasets/MegaDepth/SceneInfo/disk_dataset.json
```

### ScanNet

Download the entire ScanNet release using the script provided by the authors and place it in `$PATHTODATASETS/datasets/ScanNet`.

We use the code from the [ScanNet repository](https://github.com/ScanNet/ScanNet) for processing ScanNet: 

```shell
cd source/datasets/scannet/preprocessing/base
sh unpack.sh
```

You can download the data splits that we use from [here](https://drive.google.com/file/d/1NMZGz15Px0g_Rw5EJj5LM_O2L_7g-PBo/view?usp=sharing) and put them in the `ScanNet` directory, or you can generate the splits yourself by using the code from a [Jupyter notebook](notebooks/scannet/dataset.ipynb). 

## Models

We provide checkpoints in the format that is compatible with the pipeline for all models used in the evaluation (excluding ablations). Download the checkpoints from [here](https://drive.google.com/file/d/1CMEA2PuzhvhpmVub05N07wPNo7Vytnxu/view?usp=sharing) and unpack it in the root of the project, i.e. as `$PATHTONESSST/runs`. Our model is located at `runs/models/shiness/checkpoints/model_r_mAA=0.7706.pt`. Checkpoints for models used in ablations will be provided on request.

We provide a standalone [script](standalone/ICCV2023/run_nessst.sh) that can run inference of NeSS-ST on an image for those who need a minimal working example.

## Scripts

We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management. Hydra commands are composed via python scripts that automate loading of required configurations. Execution of tasks is done by bash scripts that call python scripts with specified Hydra commands. 

[scripts/ICCV2023](scripts/ICCV2023) contains scripts for hyper-parameters (lowe ratio, inlier threshold) tuning and testing:
1. [Homography estimation and classical metrics of HPatches](scripts/ICCV2023/htune_test_hpatches.sh)
2. [Fundamental matrix estimation and hyper-parameters tuning on IMC-PT](scripts/ICCV2023/htune_test_imcpt.sh)
3. [Fundamental matrix estimation on MegaDepth](scripts/ICCV2023/test_megadepth.sh)
4. [Essential matrix estimation and hyper-parameters tuning on ScanNet](scripts/ICCV2023/htune_test_scannet.sh)

Additionally, we provide scripts for running ablations:
1. [Influence of thresholding on SS-ST and RS-ST](scripts/ICCV2023/htune_test_eval_params_ablation.sh)
2. [Base detector ablation](scripts/ICCV2023/train_htune_test_criterion_ablation.sh)
3. [Evaluation with different number of keypoints](scripts/ICCV2023/htune_test_num_features_ablation.sh)

And the [script](scripts/ICCV2023/train_htune_test.sh) for training, tuning and testing of models proposed in the paper.

## Results

The visualization of the results is done via Jupyter notebooks:

1. [Evaluation](notebooks/hpatches/ICCV2023/evaluation.ipynb) and [hyper-parameters tuning](notebooks/hpatches/ICCV2023/htune.ipynb) on HPatches
2. [Evaluation](notebooks/imcpt/ICCV2023/evaluation.ipynb), [hyper-parameters tuning](notebooks/imcpt/ICCV2023/htune.ipynb) and [models sizes and inference time](notebooks/imcpt/ICCV2023/models_stats.ipynb) on IMC-PT
3. [Evaluation](notebooks/megadepth/ICCV2023/evaluation.ipynb) and [examples of loss functions](notebooks/megadepth/ICCV2023/losses) on MegaDepth
4. [Evaluation](notebooks/scannet/ICCV2023/evaluation.ipynb) and [hyper-parameters](notebooks/scannet/ICCV2023/htune.ipynb) tuning on ScanNet

## Citation

```
@inproceedings{pakulev2023nessst,
  author    = {Pakulev, Konstantin and Vakhitov, Alexander and Ferrer, Gonzalo},
  title     = {NeSS-ST: Detecting Good and Stable Keypoints with a Neural Stability Score and the Shi-Tomasi detector},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {9578-9588}
}
```
