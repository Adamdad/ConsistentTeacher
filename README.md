# 🧑‍🏫 Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection 🧑‍🏫

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-100)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-100?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-10)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-10?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-2)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-2?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-5)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-5?p=consistent-teacher-provides-better-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistent-teacher-provides-better-1/semi-supervised-object-detection-on-coco-1)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-1?p=consistent-teacher-provides-better-1)

This repository contains the offical implementation for our CVPR-2023 paper. 

✨We are now able to train detector on 10% MS-COCO to 40 mAP✨

**Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection**

[[arxiv](https://arxiv.org/abs/2209.01589)] [[code](https://github.com/Adamdad/ConsistentTeacher)]

Xinjiang Wang*, Xingyi Yang*, Shilong Zhang, Yijiang Li, Litong Feng, Shijie Fang, Chengqi Lyu, Kai Chen, Wayne Zhang 


> In this paper, we systematically investigate the inconsistency problems in semi-supervised object detection, where the pseudo boxes may be highly inaccurate and vary greatly at different stages of training. To alleviate the aforementioned problem, we present a holistic semi-supervised object detector termed Consistent-Teacher. Consistent-Teacher achieves compelling improvement on a wide range of evaluations and serves as a new solid baseline for SSOD.


![](assets/pipeline.jpg)

## Main Results

**MS-COCO 1%/2%/5/%/10% Labeled Data**

| Method | Data | mAP| config| Google Drive | Baidu Drive
|---- | --- |----| ---- | -----| ----|
| ConsistentTeacher | MS-COCO 1% | 25.50 | [config](configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_1p.py) | [log](https://drive.google.com/file/d/1uxGFRCqPTILrGLDzoT2-tjuvZ-cEbGAd/view?usp=share_link) | [log](https://pan.baidu.com/s/1iJlIgiD2K_jGoHff6UCSjA?pwd=gyex)/[ckpt](https://getshogun.com/wp-content/uploads/2022/07/61d5e1b6ae8db76cba5ac2fe_Coming-Soon-Page.jpeg)
| ConsistentTeacher | MS-COCO 2% | 30.70 | [config](configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_2p.py) | [log](https://drive.google.com/file/d/1Q_1bphDNvwIXO2BYOoc-od7-RskYRgrM/view?usp=share_link)/[ckpt](https://drive.google.com/file/d/1uWKXvKyAlKaRqS_zOHGh4MCNFqXN2I0r/view?usp=share_link)| [log](https://pan.baidu.com/s/1vilhESSszMte8d_06Ok7aQ?pwd=se3y)/[ckpt](https://pan.baidu.com/s/19H4VtiiprBbYe9nqWzFYjw?pwd=x9rw)
| ConsistentTeacher | MS-COCO 5% | 36.60 | [config](configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_5p.py) | [log](https://drive.google.com/file/d/1aB7p0qWHaUo8gZDqq1Aj2NJnMFYvrrG0/view?usp=share_link)/[ckpt](https://drive.google.com/file/d/1KmoqCLui_QfCiyM2PC1FWenF1wXzwwHq/view?usp=share_link)| [log](https://pan.baidu.com/s/1akJTh7_DOHf2nQHV6UgZoA?pwd=mg98)/[ckpt](https://pan.baidu.com/s/1A9TnfFhCyB5GzCbAlCOfHg?pwd=y6x3)
| ConsistentTeacher | MS-COCO 10% | 40.20 | [config](configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_10p.py)| [log](https://drive.google.com/file/d/1JHIJUqnQF-NEcVlZq7NhbBn55ShZpOjC/view?usp=share_link)/[ckpt](https://drive.google.com/file/d/1z7NmpUXYaqDHTkfpFgXWyz6RMKqx96kD/view?usp=share_link)|[log](https://pan.baidu.com/s/1ClexLVpVd_PJJRBKutRQuQ?pwd=mwb8)/[ckpt](https://pan.baidu.com/s/15og5Nqa7xJDeFORBxEf05w?pwd=mevh)|
| ConsistentTeacher 2x8 | MS-COCO 10% | 38.00 | [config](configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_10p_2x8.py)|[log](https://drive.google.com/file/d/1dJsNrVrxPjSUNtWpUnquQ9DcCKP8jtSy/view?usp=share_link)/[ckpt](https://drive.google.com/file/d/1IrvqHg-CRaAOgB_cY0qUpXGHGx-4_uT7/view?usp=share_link) | [log](https://pan.baidu.com/s/1r24ciDb1pZL5DpA5gEe9ew?pwd=k6pn)/[ckpt](https://pan.baidu.com/s/1OW9VXYgdAiA7ao6uDV9IoQ?pwd=ybxn)
| ConsistentTeacher 2x8 (FP16)| MS-COCO 10% | 37.90 | [config](configs/consistent-teacher/consistent_teacher_r50_fpn_coco_180k_10p_2x8_fp16.py)|[log](https://drive.google.com/file/d/1_i0RcKSwnDubq4MQa13K_E5-hkbkCq0G/view?usp=share_link) | [log](https://pan.baidu.com/s/1dOZdMjMdrrHQ4l-SZbPa_A?pwd=3pi4)[ckpt](https://getshogun.com/wp-content/uploads/2022/07/61d5e1b6ae8db76cba5ac2fe_Coming-Soon-Page.jpeg)

**MS-COCO100% Labeled + Unlabeled Data**

| Method | Data | mAP| config| Google Drive | Baidu Drive
|---- | ----| ---- |-----| ----| -----|
| ConsistentTeacher 5x8 | MS-COCO 100% + unlabeled |48.20 | [config](configs/consistent-teacher/consistent_teacher_r50_fpn_coco_360k_fulldata.py)|[log](https://drive.google.com/file/d/12-4CKRBigOa7T4vaQRYznm5PfxtuiOBU/view?usp=share_link)/[ckpt](https://drive.google.com/file/d/1d3BUx6NrarkcTtWoK4O6OAxr1DYW-hse/view?usp=share_link)| [log](https://pan.baidu.com/s/18Uly6NyujoZU1GYJtxcK7g?pwd=s7td)/[ckpt](https://pan.baidu.com/s/1D7FxZQY33woNXtpq_CnzpQ?pwd=nws3)

**PASCAL VOC07 Label + VOC12 Unlabel**

| Method | Data| mAP| AP50| config| 
|---- | ----| -----| ---- | ---- |
| ConsistentTeacher |PASCAL VOC07 Label + VOC12 Unlabel| 59.00 | 81.00 |  [config](configs/consistent-teacher/consistent_teacher_r50_fpn_voc0712_72k.py)|

## File Orgnizations

```
├── configs              
    ├── baseline
    │   |-- mean_teacher_retinanet_r50_fpn_coco_180k_10p.py       
    |       # Mean Teacher COCO 10% config
    |   |-- mean_teacher_retinanet_r50_fpn_voc0712_72k.py      
    |       # Mean Teacher VOC0712 config
    ├── consistent-teacher
    |   |-- consistent_teacher_r50_fpn_coco_360k_fulldata.py           
    |       # Consistent Teacher COCO label+unlabel config
    |
    |   |-- consistent_teacher_r50_fpn_coco_180k_1/2/5/10p.py           
    |       # Consistent Teacher COCO 1%/2%/5%/10% config
    |   |-- consistent_teacher_r50_fpn_coco_180k_10p_2x8.py     
    |       # Consistent Teacher COCO 10% config with 8x2 GPU
    |   |-- consistent_teacher_r50_fpn_voc0712_72k.py             
    |       # Consistent Teacher VOC0712 config
├── ssod
    |-- models/mean_teacher.py                           
    |   # Consistent Teacher Class file
    |-- models/consistent_teacher.py                     
    |   # Consistent Teacher Class file
    |-- models/dense_heads/fam3d.py                      
    |   # FAM-3D Class file
    |-- models/dense_heads/improved_retinanet.py                      
    |   # ImprovedRetinaNet baseline file
    |-- core/bbox/assigners/dynamic_assigner.py
    |   # Aadaptive Sample Assignment Class file
├── tools
    |-- dataset/semi_coco.py
    |   # COCO data preprocessing
    |-- train.py/test.py
    |   # Main file for train and evaluate the models

```
### Notes
- Defaultly, all models are trained on 8*V100 GPUs with 5 images per GPU.

## Usage

### Requirements
- `Pytorch=1.9.0`
- `mmdetection=2.25.0`
- `mmcv=1.3.9`
- `wandb=0.10.31`

or 

- `mmdetection=2.28.1`
- `mmcv=1.7.1`

#### Notes
- We use [wandb](https://wandb.ai/) for visualization, if you don't want to use it, just comment line `273-284` in `configs/consistent-teacher/base.py`.

### Installation
Install all the requirements [INSTALL](https://mmdetection.readthedocs.io/en/stable/get_started.html), then git pull the `mmdetecton` repo and ConsistentTeacher under the same folder
```
git clone https://github.com/open-mmlab/mmdetection.git
git clone https://github.com/Adamdad/ConsistentTeacher.git
cd ConsistentTeacher/
pip install -e .
```


### Data Preparation
#### COCO Dataset
- Download the COCO dataset
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA should be a directory contains coco dataset.
# For eg.:
# YOUR_DATA/
#  coco_semi/
#     instances_train2017.${fold}@${percent}.json
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

- To train model on the **partial labeled data** and **full labeled data** setting:

```shell script
# CONFIG_FILE_PATH: the config file for experiment.
# GPU_NUM: number of gpus to run the job
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

## License

This project is released under the [Apache 2.0 license](LICENCE).

## Citation

```bibtex
@article{wang2023consistent,
    author    = {Xinjiang Wang, Xingyi Yang, Shilong Zhang, Yijiang Li, Litong Feng, Shijie Fang, Chengqi Lyu, Kai Chen, Wayne Zhang },
    title     = {Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection},
    journal   = {The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
    year      = {2023},
}
```
