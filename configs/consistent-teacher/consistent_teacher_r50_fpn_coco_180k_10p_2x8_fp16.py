_base_ = './consistent_teacher_r50_fpn_coco_180k_10p_2x8.py'

fp16 = dict(loss_scale=512.)