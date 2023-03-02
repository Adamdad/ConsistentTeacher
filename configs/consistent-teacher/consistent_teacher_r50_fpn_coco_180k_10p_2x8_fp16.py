_base_ = './consistent_teacher_r50_fpn_coco_180k_10p.py'

fp16 = dict(loss_scale=512.)