# BEVFormer-Pico (student with ConvNeXt-Nano) with Knowledge Distillation from BEVFormer-Tiny
# This config inherits from bevformer_tiny and modifies the model architecture to be smaller

_base_ = [
    '../bevformer/bevformer_tiny.py',  # Inherit basic settings, dataset, etc.
]

# --- Student Model: BEVFormer-Pico/Nano ---
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

# 学生模型权重预加载
student_checkpoint_path = 'work_dirs/bevformer_pico_distill/latest.pth'

# --- Distillation Configuration ---
# Load student model checkpoint if available for resuming training
load_from = student_checkpoint_path
resume_from = student_checkpoint_path

distillation_cfg = dict(
    loss_weight=1.0,       # Weight for the BEV feature distillation loss
    loss_type='mse',       # 'mse' or 'l1'
    # Type of adapter if channel dims mismatch ('conv1x1', 'linear', or None)
    adapter='conv1x1',
                           # Adapter will be used if student_embed_dims != teacher_embed_dims
)

# --- Define Student Model using BEVFormerDistill ---
# Note: convnext_nano stage 3 (out_indices=(3,)) output channels are 640 according to timm.
# If bevformer_tiny.py img_neck expects 2048 (from ResNet50), this needs careful handling.
# Here, we define the student's img_neck to take convnext_nano's output.

# Channel number for convnext_nano output at out_indices=(3,)
# This corresponds to the output of the 4th stage (index 3)
CONVNEXT_NANO_STAGE3_CHANNELS = 640

model = dict(
    type='BEVFormerDistill',  # Our new distillation model class
    # Teacher model and distillation specific configurations
    teacher_cfg_path=teacher_cfg_path,
    teacher_checkpoint_path=teacher_checkpoint_path,
    distillation_cfg=distillation_cfg,

    # Student BEVFormer (using ConvNeXt-Nano backbone)
    use_grid_mask=True,  # Copied from tiny
    video_test_mode=True,  # Copied from tiny
    img_backbone=dict(
        _delete_=True,  # Prevent merging with base config's img_backbone
        type='ConvNeXtTimm',
        # Smaller backbone for pico (假设有这个模型，或使用其他小型模型)
        model_name='convnext_nano',
        pretrained=True,
        out_indices=(3,),  # Typically last stage
        drop_path_rate=0.1,  # Adjusted
        layer_scale_init_value=1e-6,
    ),
    img_neck=dict(
        type='FPN',
        # ConvNeXt-Femto stage 3 output channels (假设值，需要根据实际模型调整)
        # Adjust based on convnext_femto actual output channels
        in_channels=[CONVNEXT_NANO_STAGE3_CHANNELS],
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

# 工作目录
work_dir = './work_dirs/bevformer_nano_student_distill_from_tiny'
