# Configuration for Standalone Fine-tuning of BEVFormer-Pico
# This config trains the student model (ConvNeXt-Nano backbone)
# using weights potentially pre-trained via distillation.

_base_ = [
    # Inherit dataset, default runtime, etc.
    '../bevformer/bevformer_tiny.py',
    # Note: bevformer_tiny.py defines its own model,
    # which we will fully override below.
]

# --- Basic Parameters (consistent with pico distillation config) ---
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# voxel_size is defined in bevformer_tiny.py and inherited.
# Ensure it's [0.2, 0.2, 8] if not. For clarity, we can redefine if needed.
# Explicitly define for clarity and to satisfy linter
voxel_size = [0.2, 0.2, 8]

# --- Student Model Parameters (BEVFormer-Pico/Nano) ---
_dim_ = 128  # Student model BEV dimension
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1  # Using single scale features from img_neck for BEVFormer encoder
bev_h_ = 32   # Smaller BEV height for student
bev_w_ = 32   # Smaller BEV width for student

# Channel number for convnext_nano output at out_indices=(3,)
CONVNEXT_NANO_STAGE3_CHANNELS = 640  # As defined in the distillation config

# --- Model Definition (Student: BEVFormer-Pico with ConvNeXt-Nano) ---
model = dict(
    type='BEVFormer',  # Standard BEVFormer model type
    use_grid_mask=True,
    video_test_mode=True,  # Kept from tiny/distill config
    # The following pretrained is for the whole BEVFormer model. Since img_backbone handles its own,
    # this might need to be set to None if it causes issues, or if not used.
    # pretrained=None, # Consider this if errors persist after fixing img_backbone
    img_backbone=dict(
        _delete_=True,               # Add this to completely discard base img_backbone config
        type='ConvNeXtTimm',
        # Specify the student's backbone (e.g., nano, femto, pico)
        model_name='convnext_nano',
        # Load ImageNet pretrained weights for the specified ConvNeXt model
        pretrained=True,
        # Output from the last stage (or specified stages)
        out_indices=(3,),
        drop_path_rate=0.1,         # Specific to ConvNeXt
        layer_scale_init_value=1e-6,  # Specific to ConvNeXt
    ),
    img_neck=dict(
        type='FPN',
        # Input from ConvNeXt-Nano's stage 3
        in_channels=[CONVNEXT_NANO_STAGE3_CHANNELS],
        out_channels=_dim_,          # Output to student's BEV dimension
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,       # Single scale output
        relu_before_extra_convs=True
    ),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=450,        # Reduced queries for pico (tiny has 900)
        num_classes=10,       # NuScenes classes
        in_channels=_dim_,    # Student's dimension
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,   # Standard BEVFormer is not two-stage here
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,  # Reduced encoder layers for pico (tiny has 3)
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
                num_layers=3,  # Reduced decoder layers for pico (tiny has 6)
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        # Reduced heads for pico (tiny has 8)
                        dict(type='MultiheadAttention',
                             embed_dims=_dim_, num_heads=4, dropout=0.1),
                        dict(type='CustomMSDeformableAttention',
                             embed_dims=_dim_, num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm',
                                     'cross_attn', 'norm', 'ffn', 'norm'),
                    # Explicitly define ffn_cfgs for decoder layers if needed to match _dim_
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,  # ensure FFN inside decoder uses student's _dim_
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,  # from distill config
                        ffn_drop=0.1,  # from distill config
                        # from distill config
                        act_cfg=dict(type='ReLU', inplace=True)
                    ),
                    # These were in distill config, ensure they are suitable
                    # Activation for decoder layer itself
                    act_cfg=dict(type='ReLU', inplace=True),
                    # Norm for decoder layer itself
                    norm_cfg=dict(type='LN'),
                    # Num FCs in FFN (consistent with ffn_cfgs.num_fcs)
                    ffn_num_fcs=2
                )
            ),
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,  # Inherited from bevformer_tiny.py or explicitly set
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
        # GIoU loss might be 0 if not primary
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)
    ),
    # Train Cfg (copied from bevformer_tiny and pico_distill, voxel_size should be available)
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],  # Standard grid size
        voxel_size=voxel_size,  # Inherited or explicitly set
        point_cloud_range=point_cloud_range,
        out_size_factor=4,  # Standard for BEVFormer
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Set to 0 if not used
            pc_range=point_cloud_range
        )
    ))
)

# --- Dataset and Dataloader Overrides (if any, beyond bev_size) ---
# bevformer_tiny.py defines data pipelines, dataset_type, data_root etc.
# We only need to ensure bev_size is correct for the student model.
data = dict(
    # samples_per_gpu and workers_per_gpu are typically inherited from bevformer_tiny
    # samples_per_gpu=1, # As in bevformer_tiny
    # workers_per_gpu=4, # As in bevformer_tiny
    train=dict(
        bev_size=(bev_h_, bev_w_)  # Override bev_size for student
        # Other params like type, data_root, ann_file, pipeline, classes, modality
        # are inherited from _base_ (bevformer_tiny.py)
    ),
    val=dict(
        bev_size=(bev_h_, bev_w_)  # Override bev_size for student
    ),
    test=dict(
        bev_size=(bev_h_, bev_w_)  # Override bev_size for student
    )
)

# --- Optimizer and LR Scheduler (specific to student fine-tuning) ---
optimizer = dict(
    type='AdamW',
    lr=1e-4,  # LR used for pico distillation, good starting point
    paramwise_cfg=dict(
        custom_keys={
            # Standard fine-tuning for backbone
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01
)

# optimizer_config is inherited from bevformer_tiny.py (grad_clip)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


# Learning policy (copied from pico distillation config)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# --- Training Schedule ---
total_epochs = 100  # As per your recent change in distillation config
# Evaluate every 3 epochs. The pipeline for evaluation will be taken from cfg.data.val.pipeline.
evaluation = dict(interval=3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# --- Runtime Settings ---
# Log, checkpoint, and other runtime settings are largely inherited from default_runtime.py via bevformer_tiny.py
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)  # Save checkpoint every epoch

# --- Pre-trained Weights for Fine-tuning ---
# Load weights from the student model trained via distillation.
# Update this path to your actual distilled student model checkpoint.
load_from = 'work_dirs/bevformer_pico_distill/latest.pth'

# resume_from can be set if you want to resume this specific fine-tuning run
# or path to a checkpoint in work_dir below
resume_from = ' work_dirs/bevformer_pico_distill_train/latest.pth'

# --- Workspace ---
work_dir = './work_dirs/bevformer_pico_standalone_ft'

# If custom imports are needed for ConvNeXtTimm or other plugins not covered by bevformer_tiny's plugin_dir
# custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False) # Example if Timm requires mmcls
# Ensure plugin=True and plugin_dir are set correctly if needed (usually inherited from bevformer_tiny.py)
# plugin = True
# plugin_dir = 'projects/mmdet3d_plugin/'
