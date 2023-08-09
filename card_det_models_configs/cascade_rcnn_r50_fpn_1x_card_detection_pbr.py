model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=213),
            dict(type='Shared2FCBBoxHead', num_classes=213),
            dict(type='Shared2FCBBoxHead', num_classes=213)
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = '/workspace/fast_data_1/data/synth_cards/pbr_cards/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/workspace/fast_data_1/data/synth_cards/pbr_cards/coco_annotations_bbox.json',
        img_prefix='/workspace/fast_data_1/data/synth_cards/pbr_cards/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=(
            'big', 'old', 'short', 'small', 'strong', 'tall', 'weak', 'young',
            'bird', 'butterfly', 'cat', 'chicken', 'cow', 'crab', 'crocodile',
            'deer', 'dog', 'duck', 'elephant', 'fish', 'frog', 'horse', 'lion',
            'monkey', 'pig', 'shell', 'snake', 'tiger', 'comb', 'shampoo',
            'shower', 'soap', 'tissue', 'toilet', 'toothbrush', 'toothpaste',
            'towel', 'alarmclock', 'bed', 'blanket', 'curtain', 'dresser',
            'lamp', 'mirror', 'pajamas', 'pillow', 'wardrobe', 'arm', 'chin',
            'ear', 'elbow', 'eye', 'eyebrow', 'foot', 'hand', 'knee', 'mouth',
            'neck', 'nose', 'shoulder', 'teeth', 'tongue', 'black', 'blue',
            'brown', 'gray', 'green', 'pink', 'red', 'white', 'yellow', 'belt',
            'cap', 'ring', 'shirt', 'shoes', 'shorts', 'skirt', 'socks',
            'umbrella', 'watch', 'friday', 'monday', 'saturday', 'sunday',
            'thursday', 'tuesday', 'wednesday', 'bread', 'cake', 'cookies',
            'doughnut', 'eggs', 'icecream', 'milk', 'noodles', 'popcorn',
            'rice', 'tofu', 'water', 'apple', 'banana', 'grape', 'mango',
            'mangosteen', 'orange', 'papaya', 'pineapple', 'rambutan',
            'roseapple', 'watermelon', 'broom', 'bucket', 'chair', 'clock',
            'door', 'fan', 'mop', 'radio', 'refrigerator', 'table',
            'telephone', 'television', 'window', 'blender', 'bowl', 'fork',
            'glass', 'knife', 'pan', 'pitcher', 'plate', 'spoon', 'april',
            'august', 'december', 'february', 'january', 'july', 'june',
            'march', 'may', 'november', 'october', 'september', 'cloud',
            'flower', 'moon', 'mountain', 'rain', 'rainbow', 'river', 'rock',
            'sand', 'thunder', 'eight', 'five', 'four', 'nine', 'one', 'seven',
            'six', 'ten', 'three', 'two', 'chef', 'doctor', 'farmer',
            'merchant', 'monk', 'nurse', 'policeman', 'soldier', 'teacher',
            'worker', 'circle', 'heptagon', 'hexagon', 'octagon', 'oval',
            'pentagon', 'rectangle', 'triangle', 'backpack', 'book', 'eraser',
            'glue', 'notebook', 'pen', 'pencil', 'ruler', 'scissors',
            'airplane', 'bicycle', 'boat', 'bus', 'car', 'helicopter',
            'motorcycle', 'train', 'truck', 'tuktuk', 'brushmyteeth', 'dress',
            'drink', 'eat', 'rubbodydry', 'sleep', 'takeabath', 'takeapee',
            'takeapoo', 'washmyhair', 'washmyhand')),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/workspace/fast_data_1/data/synth_cards/cardBox/robot_card_test_coco.json',
        img_prefix='/workspace/fast_data_1/data/synth_cards/cardBox',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=(
            'big', 'old', 'short', 'small', 'strong', 'tall', 'weak', 'young',
            'bird', 'butterfly', 'cat', 'chicken', 'cow', 'crab', 'crocodile',
            'deer', 'dog', 'duck', 'elephant', 'fish', 'frog', 'horse', 'lion',
            'monkey', 'pig', 'shell', 'snake', 'tiger', 'comb', 'shampoo',
            'shower', 'soap', 'tissue', 'toilet', 'toothbrush', 'toothpaste',
            'towel', 'alarmclock', 'bed', 'blanket', 'curtain', 'dresser',
            'lamp', 'mirror', 'pajamas', 'pillow', 'wardrobe', 'arm', 'chin',
            'ear', 'elbow', 'eye', 'eyebrow', 'foot', 'hand', 'knee', 'mouth',
            'neck', 'nose', 'shoulder', 'teeth', 'tongue', 'black', 'blue',
            'brown', 'gray', 'green', 'pink', 'red', 'white', 'yellow', 'belt',
            'cap', 'ring', 'shirt', 'shoes', 'shorts', 'skirt', 'socks',
            'umbrella', 'watch', 'friday', 'monday', 'saturday', 'sunday',
            'thursday', 'tuesday', 'wednesday', 'bread', 'cake', 'cookies',
            'doughnut', 'eggs', 'icecream', 'milk', 'noodles', 'popcorn',
            'rice', 'tofu', 'water', 'apple', 'banana', 'grape', 'mango',
            'mangosteen', 'orange', 'papaya', 'pineapple', 'rambutan',
            'roseapple', 'watermelon', 'broom', 'bucket', 'chair', 'clock',
            'door', 'fan', 'mop', 'radio', 'refrigerator', 'table',
            'telephone', 'television', 'window', 'blender', 'bowl', 'fork',
            'glass', 'knife', 'pan', 'pitcher', 'plate', 'spoon', 'april',
            'august', 'december', 'february', 'january', 'july', 'june',
            'march', 'may', 'november', 'october', 'september', 'cloud',
            'flower', 'moon', 'mountain', 'rain', 'rainbow', 'river', 'rock',
            'sand', 'thunder', 'eight', 'five', 'four', 'nine', 'one', 'seven',
            'six', 'ten', 'three', 'two', 'chef', 'doctor', 'farmer',
            'merchant', 'monk', 'nurse', 'policeman', 'soldier', 'teacher',
            'worker', 'circle', 'heptagon', 'hexagon', 'octagon', 'oval',
            'pentagon', 'rectangle', 'triangle', 'backpack', 'book', 'eraser',
            'glue', 'notebook', 'pen', 'pencil', 'ruler', 'scissors',
            'airplane', 'bicycle', 'boat', 'bus', 'car', 'helicopter',
            'motorcycle', 'train', 'truck', 'tuktuk', 'brushmyteeth', 'dress',
            'drink', 'eat', 'rubbodydry', 'sleep', 'takeabath', 'takeapee',
            'takeapoo', 'washmyhair', 'washmyhand')),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/workspace/fast_data_1/data/synth_cards/cardBox/robot_card_test_coco.json',
        img_prefix='/workspace/fast_data_1/data/synth_cards/cardBox',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=(
            'big', 'old', 'short', 'small', 'strong', 'tall', 'weak', 'young',
            'bird', 'butterfly', 'cat', 'chicken', 'cow', 'crab', 'crocodile',
            'deer', 'dog', 'duck', 'elephant', 'fish', 'frog', 'horse', 'lion',
            'monkey', 'pig', 'shell', 'snake', 'tiger', 'comb', 'shampoo',
            'shower', 'soap', 'tissue', 'toilet', 'toothbrush', 'toothpaste',
            'towel', 'alarmclock', 'bed', 'blanket', 'curtain', 'dresser',
            'lamp', 'mirror', 'pajamas', 'pillow', 'wardrobe', 'arm', 'chin',
            'ear', 'elbow', 'eye', 'eyebrow', 'foot', 'hand', 'knee', 'mouth',
            'neck', 'nose', 'shoulder', 'teeth', 'tongue', 'black', 'blue',
            'brown', 'gray', 'green', 'pink', 'red', 'white', 'yellow', 'belt',
            'cap', 'ring', 'shirt', 'shoes', 'shorts', 'skirt', 'socks',
            'umbrella', 'watch', 'friday', 'monday', 'saturday', 'sunday',
            'thursday', 'tuesday', 'wednesday', 'bread', 'cake', 'cookies',
            'doughnut', 'eggs', 'icecream', 'milk', 'noodles', 'popcorn',
            'rice', 'tofu', 'water', 'apple', 'banana', 'grape', 'mango',
            'mangosteen', 'orange', 'papaya', 'pineapple', 'rambutan',
            'roseapple', 'watermelon', 'broom', 'bucket', 'chair', 'clock',
            'door', 'fan', 'mop', 'radio', 'refrigerator', 'table',
            'telephone', 'television', 'window', 'blender', 'bowl', 'fork',
            'glass', 'knife', 'pan', 'pitcher', 'plate', 'spoon', 'april',
            'august', 'december', 'february', 'january', 'july', 'june',
            'march', 'may', 'november', 'october', 'september', 'cloud',
            'flower', 'moon', 'mountain', 'rain', 'rainbow', 'river', 'rock',
            'sand', 'thunder', 'eight', 'five', 'four', 'nine', 'one', 'seven',
            'six', 'ten', 'three', 'two', 'chef', 'doctor', 'farmer',
            'merchant', 'monk', 'nurse', 'policeman', 'soldier', 'teacher',
            'worker', 'circle', 'heptagon', 'hexagon', 'octagon', 'oval',
            'pentagon', 'rectangle', 'triangle', 'backpack', 'book', 'eraser',
            'glue', 'notebook', 'pen', 'pencil', 'ruler', 'scissors',
            'airplane', 'bicycle', 'boat', 'bus', 'car', 'helicopter',
            'motorcycle', 'train', 'truck', 'tuktuk', 'brushmyteeth', 'dress',
            'drink', 'eat', 'rubbodydry', 'sleep', 'takeabath', 'takeapee',
            'takeapoo', 'washmyhair', 'washmyhand')))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 4])
runner = dict(type='EpochBasedRunner', max_epochs=5)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained_models/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
classes = (
    'big', 'old', 'short', 'small', 'strong', 'tall', 'weak', 'young', 'bird',
    'butterfly', 'cat', 'chicken', 'cow', 'crab', 'crocodile', 'deer', 'dog',
    'duck', 'elephant', 'fish', 'frog', 'horse', 'lion', 'monkey', 'pig',
    'shell', 'snake', 'tiger', 'comb', 'shampoo', 'shower', 'soap', 'tissue',
    'toilet', 'toothbrush', 'toothpaste', 'towel', 'alarmclock', 'bed',
    'blanket', 'curtain', 'dresser', 'lamp', 'mirror', 'pajamas', 'pillow',
    'wardrobe', 'arm', 'chin', 'ear', 'elbow', 'eye', 'eyebrow', 'foot',
    'hand', 'knee', 'mouth', 'neck', 'nose', 'shoulder', 'teeth', 'tongue',
    'black', 'blue', 'brown', 'gray', 'green', 'pink', 'red', 'white',
    'yellow', 'belt', 'cap', 'ring', 'shirt', 'shoes', 'shorts', 'skirt',
    'socks', 'umbrella', 'watch', 'friday', 'monday', 'saturday', 'sunday',
    'thursday', 'tuesday', 'wednesday', 'bread', 'cake', 'cookies', 'doughnut',
    'eggs', 'icecream', 'milk', 'noodles', 'popcorn', 'rice', 'tofu', 'water',
    'apple', 'banana', 'grape', 'mango', 'mangosteen', 'orange', 'papaya',
    'pineapple', 'rambutan', 'roseapple', 'watermelon', 'broom', 'bucket',
    'chair', 'clock', 'door', 'fan', 'mop', 'radio', 'refrigerator', 'table',
    'telephone', 'television', 'window', 'blender', 'bowl', 'fork', 'glass',
    'knife', 'pan', 'pitcher', 'plate', 'spoon', 'april', 'august', 'december',
    'february', 'january', 'july', 'june', 'march', 'may', 'november',
    'october', 'september', 'cloud', 'flower', 'moon', 'mountain', 'rain',
    'rainbow', 'river', 'rock', 'sand', 'thunder', 'eight', 'five', 'four',
    'nine', 'one', 'seven', 'six', 'ten', 'three', 'two', 'chef', 'doctor',
    'farmer', 'merchant', 'monk', 'nurse', 'policeman', 'soldier', 'teacher',
    'worker', 'circle', 'heptagon', 'hexagon', 'octagon', 'oval', 'pentagon',
    'rectangle', 'triangle', 'backpack', 'book', 'eraser', 'glue', 'notebook',
    'pen', 'pencil', 'ruler', 'scissors', 'airplane', 'bicycle', 'boat', 'bus',
    'car', 'helicopter', 'motorcycle', 'train', 'truck', 'tuktuk',
    'brushmyteeth', 'dress', 'drink', 'eat', 'rubbodydry', 'sleep',
    'takeabath', 'takeapee', 'takeapoo', 'washmyhair', 'washmyhand')
fp16 = dict(loss_scale=512.0)
work_dir = './work_dirs/cascade_rcnn_r50_fpn_1x_card_detection_pbr'
auto_resume = False
gpu_ids = [0]
