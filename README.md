# Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-100)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-100?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-10)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-10?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-2)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-2?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-5)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-5?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-1)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-1?p=consistent-teacher-provides-better-1)

This repository contains the offical implementation for our paper

**Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection**

[arxiv](https://arxiv.org/abs/2209.01589) [code](https://github.com/Adamdad/ConsistentTeacher)

Xinjiang Wang*, Xingyi Yang*, Shilong Zhang, Yijiang Li, Litong Feng, Shijie Fang, Chengqi Lyu, Kai Chen, Wayne Zhang 





## File Orgnizations

```
  configs              
  - baseline
    - mean_teacher_retinanet_r50_fpn_coco_180k_10p.py       
      # Mean Teacher COCO 10% config
    - mean_teacher_retinanet_r50_fpn_voc0712_72k.py      
      # Mean Teacher VOC0712 config

  - consistent-teacher
    - consistent_teacher_r50_fpn_coco_180k_10p.py           
      # Consistent Teacher COCO 10% config
    - consistent_teacher_r50_fpn_voc0712_72k.py             
      # Consistent Teacher VOC0712 config


  ssod
  - models/mean_teacher.py                           
    # Consistent Teacher Class file
  - models/consistent_teacher.py                     
    # Consistent Teacher Class file
  - models/dense_heads/fam3d.py                      
    # FAM-3D Class file
  - models/dense_heads/improved_retinanet.py                      
    #  ImprovedRetinaNet baseline file
  - core/bbox/assigners/dynamic_assigner.py
    # Aadaptive Sample Assignment Class file

  tools
  - dataset/semi_coco.py
    # COCO data preprocessing
  - train.py/test.py
    # Main file for train and evaluate the models

```
### Notes
- All models are trained on 8*V100 GPUs with 5 images per GPU.

## Usage

### Requirements
- `Ubuntu 16.04`
- `Anaconda3` with `python=3.6`
- `Pytorch=1.9.0`
- `mmdetection=2.25.0`
- `mmcv=1.3.9`
- `wandb=0.10.31`

#### Notes
- We use [wandb](https://wandb.ai/) for visualization, if you don't want to use it, just comment line `273-284` in `configs/consistent-teacher/base.py`.

### Installation
```
make install
```

### Data Preparation
#### COCO Dataset
- Download the COCO dataset
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA should be a directory contains coco dataset.
# For eg.:
# YOUR_DATA/
#  coco/
#     train2017/
#     val2017/
#     unlabeled2017/
#     annotations/
ln -s ${YOUR_DATA} data
bash tools/dataset/prepare_coco_data.sh conduct

```
For concrete instructions of what should be downloaded, please refer to `tools/dataset/prepare_coco_data.sh` line [`11-24`](https://github.com/microsoft/SoftTeacher/blob/863d90a3aa98615be3d156e7d305a22c2a5075f5/tools/dataset/prepare_coco_data.sh#L11)
### Training

- To train model on the **partial labeled data** setting:
```shell script
# JOB_TYPE: 'baseline' or 'semi', decide which kind of job to run
# PERCENT_LABELED_DATA: 1, 5, 10. The ratio of labeled coco data in whole training dataset.
# GPU_NUM: number of gpus to run the job
for FOLD in 1 2 3 4 5;
do
  bash tools/dist_train_partially.sh <JOB_TYPE> ${FOLD} <PERCENT_LABELED_DATA> <GPU_NUM>
done
```
For example, we could run the following scripts to train our model on 10% labeled data with 8 GPUs:

```shell script
for FOLD in 1 2 3 4 5;
do
  bash tools/dist_train_partially.sh semi ${FOLD} 10 8
done
```

- To train model on the **full labeled data** setting:

```shell script
bash tools/dist_train.sh <CONFIG_FILE_PATH> <NUM_GPUS>
```
For example, to train ours `R50` model with 8 GPUs:
```shell script
bash tools/dist_train.sh configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_10p.py 8
```
- To train model on **new dataset**:

The core idea is to convert a new dataset to coco format. Details about it can be found in the [adding new dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/customize_dataset.md).

#### VOC0712 Dataset
- Download JSON files for unlabeled images PASCAL VOC data in COCO format
```
cd ${DATAROOT}

wget https://storage.cloud.google.com/gresearch/ssl_detection/STAC_JSON.tar
tar -xf STAC_JSON.tar.gz
# voc/VOCdevkit/VOC2007/instances_test.json
# voc/VOCdevkit/VOC2007/instances_trainval.json
# voc/VOCdevkit/VOC2012/instances_trainval.json
```

