mmdet_base = "../../../mmdetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/datasets/voc0712.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
]

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='FAM3DHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_type='anchor_based',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13, iou_factor=2.0),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1000, 480), (1000, 800)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1000, 480), (1000, 800)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1000, 480), (1000, 800)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_teacher=strong_pipeline, unsup_student=weak_pipeline
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


CLASSES = ('train', 'car', 'cat', 'chair', 'person', 'diningtable', 'sofa', 
           'horse', 'bicycle', 'bird', 'cow', 'aeroplane', 'tvmonitor', 
           'bottle', 'pottedplant', 'boat', 'sheep', 'bus', 'motorbike', 'dog')

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type="CocoDataset",
            ann_file="data/VOCdevkit/VOC2007/instances_trainval.json",
            img_prefix="data/VOCdevkit/VOC2007/JPEGImages",
            pipeline=train_pipeline,
            classes=CLASSES,
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/VOCdevkit/VOC2012/instances_trainval.json",
            img_prefix="data/VOCdevkit/VOC2012/JPEGImages",
            pipeline=unsup_pipeline,
            classes=CLASSES,
            filter_empty_gt=False,
        ),
    ),
    val=dict(
        type="CocoDataset",
        pipeline=test_pipeline,
        img_prefix="data/VOCdevkit/VOC2007/JPEGImages",
        ann_file="data/VOCdevkit/VOC2007/instances_test.json",
        classes=CLASSES),
    test=dict(
        type="CocoDataset",
        pipeline=test_pipeline,
        img_prefix="data/VOCdevkit/VOC2007/JPEGImages",
        ann_file="data/VOCdevkit/VOC2007/instances_test.json",
        classes=CLASSES),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
            by_prob=True,
            # at_least_one=True,
            epoch_length=7330,
        )
    ),
)

semi_wrapper = dict(
    type="ConsistentTeacher",
    model="${model}",
    train_cfg=dict(
        num_scores=100,
        dynamic_ratio=1.0,
        warmup_step=1000,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
    ),
    test_cfg=dict(inference_on="teacher"),
)


custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type='SetIterInfoHook'),
    dict(type="MeanTeacher", momentum=0.9995, interval=1, warm_up=0),
]
# evaluation = dict(type="SubModulesDistEvalHook", interval=4000)
evaluation = dict(type="SubModulesDistEvalHook", metric='bbox', interval=4000)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(step=[72000, 72000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=72000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=2)

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
                    # fold="${fold}",
                    # percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        )

    ],
)
fp16 = None

