# Configuration for Standalone Fine-tuning of BEVFormer-Pico with SE-ResNet-18
# This config trains the student model (SE-ResNet-18 backbone)
# using weights pre-trained via our distillation setup.

_base_ = [
    # Inherit dataset, default runtime, etc.
    '../bevformer/bevformer_tiny.py',
]

# --- Basic Parameters (consistent with pico distillation config) ---
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# --- Student Model Parameters (BEVFormer-Pico/Nano scale) ---
_dim_ = 128
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 32
bev_w_ = 32

# Channel number for seresnet18 output at out_indices=(3,)
SERESNET18_STAGE3_CHANNELS = 256

# --- Model Definition (Student: BEVFormer-Pico with SE-ResNet-18) ---
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        _delete_=True,
        type='TimmBackbone',
        model_name='seresnet18',
        pretrained=True,  # Start with ImageNet pre-training for the backbone
        out_indices=(3,),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[SERESNET18_STAGE3_CHANNELS],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True
    ),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=450,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention',
                             embed_dims=_dim_, num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D', embed_dims=_dim_, num_points=8, num_levels=_num_levels_
                            ),
                            embed_dims=_dim_,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)
                    ),
                    operation_order=('self_attn', 'norm',
                                     'cross_attn', 'norm', 'ffn', 'norm'),
                    norm_cfg=dict(type='LN')
                ),
            ),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention',
                             embed_dims=_dim_, num_heads=4, dropout=0.1),
                        dict(type='CustomMSDeformableAttention',
                             embed_dims=_dim_, num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm',
                                     'cross_attn', 'norm', 'ffn', 'norm'),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)
                    ),
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'),
                    ffn_num_fcs=2
                )
            ),
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)
    ),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range
        )
    ))
)

data = dict(
    train=dict(bev_size=(bev_h_, bev_w_)),
    val=dict(bev_size=(bev_h_, bev_w_)),
    test=dict(bev_size=(bev_h_, bev_w_))
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 24
evaluation = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)

# Load weights from the student model trained via distillation.
# NOTE: The work_dir name is a legacy from previous experiments.
load_from = 'work_dirs/bevformer_nano_student_distill_from_tiny/latest.pth'

# --- Workspace ---
work_dir = './work_dirs/bevformer_seresnet18_standalone_ft'

custom_imports = dict(
    imports=['projects.mmdet3d_plugin.models.backbones.timm_backbone'],
    allow_failed_imports=False
)
