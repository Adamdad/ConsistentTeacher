_base_ = './consistent_teacher_r50_fpn_coco_180k_10p_2x8.py'

model = dict(
    backbone=dict(
        frozen_stages=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', 
                      checkpoint='/Checkpoint/yangxingyi/Pretrained/swav_800ep_pretrain.pth.tar')))