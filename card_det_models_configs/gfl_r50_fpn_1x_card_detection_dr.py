dataset_type = 'CocoDataset'
data_root = '/workspace/fast_data_1/data/synth_cards/unity_cards_dr_40k/unity_cards_dr_40k/'
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
        '/workspace/fast_data_1/data/synth_cards/unity_cards_dr_40k/unity_cards_dr_40k/coco_labels.json',
        img_prefix=
        '/workspace/fast_data_1/data/synth_cards/unity_cards_dr_40k/unity_cards_dr_40k/RGBca1a7718-5d2f-4240-b0e6-603a8c3092c1',
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
load_from = 'pretrained_models/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='GFL',
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
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=213,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
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
work_dir = './work_dirs/gfl_r50_fpn_1x_card_detection_dr'
auto_resume = False
gpu_ids = [0]
