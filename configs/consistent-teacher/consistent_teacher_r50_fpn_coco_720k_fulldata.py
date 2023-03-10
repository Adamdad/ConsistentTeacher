_base_ = './consistent_teacher_r50_fpn_coco_180k_10p.py'

model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=False)
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13, iou_factor=3.0))
)


fold = 1
percent = 100

data = dict(
    train=dict(
        sup=dict(
            ann_file="data/coco/annotations/instances_train2017.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            ann_file="data/coco_semi/semi_supervised/instances_unlabeled2017.json",
            img_prefix="data/coco/unlabeled2017/",
        ),
    ),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type='SetIterInfoHook'),
    dict(type="MeanTeacher", momentum=0.9998, interval=1, warm_up=0),
]
evaluation = dict(type="SubModulesDistEvalHook", evaluated_modules=['teacher'], interval=4000)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)


lr_config = dict(step=[480000, 640000])

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=720000)
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
fp16 = None
