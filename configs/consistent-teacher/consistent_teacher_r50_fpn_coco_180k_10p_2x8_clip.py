_base_ = './consistent_teacher_r50_fpn_coco_180k_10p_2x8.py'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
                      prefix='visual.',
                      checkpoint='/Checkpoint/yangxingyi/Pretrained/clip_RN50.pth')))
