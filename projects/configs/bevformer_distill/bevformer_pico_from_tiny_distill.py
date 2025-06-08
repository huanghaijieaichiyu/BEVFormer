# BEVFormer-Pico (student with SE-ResNet-18) with Knowledge Distillation from BEVFormer-Tiny
# This config inherits from bevformer_tiny and modifies the model architecture to be smaller

_base_ = [
    '../bevformer/bevformer_tiny.py',  # Inherit basic settings, dataset, etc.
]

# --- Student Model: BEVFormer-Pico ---
# Overrides for student version
point_cloud_range = [-51.2, -51.2, -5.0, 51.2,
                     51.2, 3.0]  # Keep same as tiny for now
_dim_ = 128  # Student model BEV dimension
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1  # Using single scale features from img_neck for BEVFormer encoder
bev_h_ = 32   # Smaller BEV height for student (tiny uses 50)
bev_w_ = 32   # Smaller BEV width for student (tiny uses 50)

# Explicitly define voxel_size here, matching bevformer_tiny.py, to resolve NameError
# This ensures it's available when the model dictionary below is processed.
# voxel_size from bevformer_tiny.py is [0.2, 0.2, 8]
voxel_size = [0.2, 0.2, 8]


# --- Teacher Model Configuration ---
teacher_cfg_path = 'projects/configs/bevformer/bevformer_tiny.py'
# 更新教师模型权重路径
teacher_checkpoint_path = 'ckpts/bevformer_tiny_epoch_24.pth'

# --- Distillation Configuration ---
# Distilling from image neck features, which are 4D and compatible with the loss function
distiller = dict(
    distill_losses=dict(
        loss_img_feat=dict(
            type='FeatureLoss',
            student_feature_loc='img_neck',  # Distill from FPN output
            teacher_feature_loc='img_neck',
            loss_func=dict(
                type='MSELoss',
                loss_weight=1.0,
            ),
            # Adapter to match student's 128 channels to teacher's 256
            channel_adapter=dict(
                type='Conv2dAdapter',
                in_channels=_dim_,
                out_channels=256,  # Teacher's FPN output dim
                kernel_size=1,
            ),
            # Spatial adapter to upsample teacher's feature map to match student's
            spatial_adapter=dict(
                type='BilinearInterpolation',
                target_size=(30, 50)  # H, W of student feature map
            )
        )
    )
)

# --- Define Student Model using DistillBEVFormer ---
model = dict(
    type='DistillBEVFormer',  # Use the implemented distiller class
    # Pass configs to the distiller wrapper
    teacher_cfg=teacher_cfg_path,
    teacher_ckpt=teacher_checkpoint_path,
    distiller=distiller,

    # --- Student BEVFormer (SE-ResNet-18 backbone) kwargs ---
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        _delete_=True,
        type='TimmBackbone',      # Use the generic timm wrapper
        model_name='seresnet18',  # Load seresnet18 from timm library
        pretrained=True,         # Use timm's pretrained weights
        out_indices=(3,),        # Output from the last stage
    ),
    img_neck=dict(
        type='FPN',
        # timm's seresnet18 with out_indices=(3,) outputs 256 channels, not 512.
        in_channels=[256],
        out_channels=_dim_,  # Student's dimension
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',  # Standard BEVFormer head
        bev_h=bev_h_,         # Student's BEV height
        bev_w=bev_w_,         # Student's BEV width
        num_query=450,        # Reduced queries for pico (tiny uses 900)
        num_classes=10,       # Same as nuscenes
        in_channels=_dim_,    # Student's dimension
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,  # Student's dimension
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,  # Reduced encoder layers for pico (tiny uses 3)
                pc_range=point_cloud_range,
                num_points_in_pillar=4,  # Copied
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
                                type='MSDeformableAttention3D', embed_dims=_dim_, num_points=8, num_levels=_num_levels_),
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
                num_layers=3,  # Reduced decoder layers for pico (tiny uses 6)
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention',
                             embed_dims=_dim_, num_heads=4, dropout=0.1),
                        dict(type='CustomMSDeformableAttention',
                             embed_dims=_dim_, num_levels=1),
                    ],
                    # DetrTransformerDecoderLayer required args
                    feedforward_channels=_ffn_dim_,  # FFN hidden dim
                    ffn_dropout=0.1,                 # FFN dropout
                    operation_order=('self_attn', 'norm',
                                     'cross_attn', 'norm', 'ffn', 'norm'),
                    # FFN config to align embed_dims
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)
                    ),
                    # Activation config
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'),                # Norm config
                    ffn_num_fcs=2                             # Number of FC layers in FFN
                )
            ),
        ),  # close PerceptionTransformer (transformer) dict
        # bbox_coder, positional_encoding, loss_cls, loss_bbox, loss_iou copied from bevformer_tiny and adjusted
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,  # Max number of detections
            voxel_size=voxel_size,  # Now uses the locally defined voxel_size
            num_classes=10),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),  # GIoU loss might be 0 if not used as primary box loss
    train_cfg=dict(pts=dict(  # Copied from bevformer_tiny
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,  # Now uses the locally defined voxel_size
        point_cloud_range=point_cloud_range,
        out_size_factor=4,  # Check consistency
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Set to 0 if not used
            pc_range=point_cloud_range)))
)


# --- Optimizer and LR Scheduler for student ---
# May need lower LR or different schedule for smaller model / distillation
optimizer = dict(
    type='AdamW',
    lr=1e-4,  # Potentially smaller LR (tiny uses 2e-4)
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# dataset settings (inherited from _base_, but need to override bev_size)
data = dict(
    train=dict(bev_size=(bev_h_, bev_w_)),
    val=dict(bev_size=(bev_h_, bev_w_)),
    test=dict(bev_size=(bev_h_, bev_w_))
)

# You might want to adjust total_epochs, evaluation interval, etc.
total_epochs = 100  # Or more, depending on how fast student learns
evaluation = dict(interval=3)  # Evaluate every epoch
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 检查点配置
checkpoint_config = dict(interval=1)

# Initialize student model with pre-trained weights to accelerate distillation.
# The runner will load weights from the teacher's checkpoint into the student model.
# Layers with mismatched names or shapes will be ignored.
load_from = 'ckpts/bevformer_tiny_epoch_24.pth'

# 工作目录
work_dir = './work_dirs/bevformer_nano_student_distill_from_tiny'

# Add custom imports to ensure the custom distiller and backbone are registered
custom_imports = dict(
    imports=[
        'projects.bevformer_mods.distillers',
        'projects.mmdet3d_plugin.models.backbones.timm_backbone'
    ],
    allow_failed_imports=False
)
