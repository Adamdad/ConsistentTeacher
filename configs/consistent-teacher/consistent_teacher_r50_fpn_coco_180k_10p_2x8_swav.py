_base_ = './consistent_teacher_r50_fpn_coco_180k_10p.py'

# fp16 = dict(_delete_=True,
#             loss_scale=512.)
model = dict(
    backbone=dict(
        # frozen_stages=0,
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
        # norm_eval=False,
        init_cfg=dict(_delete_=True,
                      type='Pretrained', 
                      checkpoint='/Checkpoint/yangxingyi/Pretrained/swav_800ep_pretrain.pth.tar')))

fold = 1
percent = 10
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            ann_file="data/coco_semi/semi_supervised/instances_train2017.${fold}@${percent}.json",
        ),
        unsup=dict(
            ann_file="data/coco_semi/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
        ),
    ),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 1],
            by_prob=False,
            # at_least_one=True,
            epoch_length=7330,
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=1.0,
    ),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type='SetIterInfoHook'),
    dict(type="MeanTeacher", momentum=0.9998, interval=1, warm_up=0),
]

optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="consistent-teacher",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        )

    ],
)


