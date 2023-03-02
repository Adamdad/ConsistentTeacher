_base_ = './consistent_teacher_r50_fpn_coco_180k_10p.py'

fold = 1
percent = 1
data = dict(
    train=dict(
        sup=dict(
            ann_file="data/coco_semi/semi_supervised/instances_train2017.${fold}@${percent}.json",
        ),
        unsup=dict(
            ann_file="data/coco_semi/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
        ),
    ),
)

log_config = dict(
    _delete_=True,
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
