optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=2,
    num_last_epochs=3,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(
                project='mmdetection',
                entity='bpninja',
                config=dict(lr=0.025, batch_size=16),
                tags=['darknet53', 'sgd', 'yolox-m', 'cpl']),
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100,
            bbox_score_thr=0.3)
    ])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=3, priority=48),
    dict(type='SyncNormHook', num_last_epochs=3, interval=1, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained_models/yolox_m_mmdet.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
img_scale = (640, 640)
model = dict(
    type='YOLOX',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.67, widen_factor=0.75),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2),
    bbox_head=dict(
        type='YOLOXHead', num_classes=213, in_channels=192, feat_channels=192),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
dataset_type = 'CocoDataset'
data_root = '/workspace/fast_data_1/data/synth_cards/cpl_dataset/'
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
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
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
            'takeapoo', 'washmyhair', 'washmyhand'),
        ann_file=
        '/workspace/fast_data_1/data/synth_cards/cpl_dataset/cpl_instances_train.json',
        img_prefix='/workspace/fast_data_1/data/synth_cards/cpl_dataset/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
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
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            classes=(
                'big', 'old', 'short', 'small', 'strong', 'tall', 'weak',
                'young', 'bird', 'butterfly', 'cat', 'chicken', 'cow', 'crab',
                'crocodile', 'deer', 'dog', 'duck', 'elephant', 'fish', 'frog',
                'horse', 'lion', 'monkey', 'pig', 'shell', 'snake', 'tiger',
                'comb', 'shampoo', 'shower', 'soap', 'tissue', 'toilet',
                'toothbrush', 'toothpaste', 'towel', 'alarmclock', 'bed',
                'blanket', 'curtain', 'dresser', 'lamp', 'mirror', 'pajamas',
                'pillow', 'wardrobe', 'arm', 'chin', 'ear', 'elbow', 'eye',
                'eyebrow', 'foot', 'hand', 'knee', 'mouth', 'neck', 'nose',
                'shoulder', 'teeth', 'tongue', 'black', 'blue', 'brown',
                'gray', 'green', 'pink', 'red', 'white', 'yellow', 'belt',
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
                'flower', 'moon', 'mountain', 'rain', 'rainbow', 'river',
                'rock', 'sand', 'thunder', 'eight', 'five', 'four', 'nine',
                'one', 'seven', 'six', 'ten', 'three', 'two', 'chef', 'doctor',
                'farmer', 'merchant', 'monk', 'nurse', 'policeman', 'soldier',
                'teacher', 'worker', 'circle', 'heptagon', 'hexagon',
                'octagon', 'oval', 'pentagon', 'rectangle', 'triangle',
                'backpack', 'book', 'eraser', 'glue', 'notebook', 'pen',
                'pencil', 'ruler', 'scissors', 'airplane', 'bicycle', 'boat',
                'bus', 'car', 'helicopter', 'motorcycle', 'train', 'truck',
                'tuktuk', 'brushmyteeth', 'dress', 'drink', 'eat',
                'rubbodydry', 'sleep', 'takeabath', 'takeapee', 'takeapoo',
                'washmyhair', 'washmyhand'),
            ann_file=
            '/workspace/fast_data_1/data/synth_cards/cpl_dataset/cpl_instances_train.json',
            img_prefix=
            '/workspace/fast_data_1/data/synth_cards/cpl_dataset/images',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
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
            'takeapoo', 'washmyhair', 'washmyhand'),
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
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
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
            'takeapoo', 'washmyhair', 'washmyhand'),
        ann_file=
        '/workspace/fast_data_1/data/synth_cards/card_eval_set_1/card_eval_set_1_coco.json',
        img_prefix=
        '/workspace/fast_data_1/data/synth_cards/card_eval_set_1/images',
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
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
fp16 = dict(loss_scale=512.0)
max_epochs = 10
num_last_epochs = 3
interval = 1
# evaluation = dict(
#     save_best='auto', interval=1, dynamic_intervals=[(7, 1)], metric='bbox')
work_dir = './work_dirs/yolox_m_8x8_300e_cpl_card_detection'
auto_resume = False
gpu_ids = [0]
