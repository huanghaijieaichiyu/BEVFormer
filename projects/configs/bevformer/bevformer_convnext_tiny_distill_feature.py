# projects/configs/bevformer/bevformer_convnext_tiny_distill_feature.py

# Inherit from the non-distilled ConvNeXt configuration
_base_ = ['./bevformer_convnext_tiny.py']

# --- Teacher Model Configuration ---
# Define the path to your pre-trained bevformer_tiny checkpoint
# !!! IMPORTANT: Replace this with the actual path to your checkpoint !!!
# !!! IMPORTANT: Verify this path !!!
teacher_ckpt = 'ckpts/bevformer_tiny_epoch_24.pth'

# Load the base configuration for the teacher model (bevformer_tiny)
# This assumes bevformer_tiny.py exists in the same directory or is findable
teacher_cfg_path = 'projects/configs/bevformer/bevformer_tiny.py'


# --- Distillation Configuration ---
model = dict(
    # Specify the type of the overall model architecture, assuming a distiller wrapper
    # This type might vary depending on your MMDetection3D/custom framework version.
    # Examples: 'DistillBEVFormer', 'GeneralDistiller'
    # If your framework handles distillation differently (e.g., via hooks or runner),
    # this structure might need adjustment.
    type='DistillBEVFormer',  # Your main distiller model wrapper

    # --- Teacher Model Setup ---
    teacher_cfg=teacher_cfg_path,
    teacher_ckpt=teacher_ckpt,
    freeze_teacher=True,  # Usually freeze the teacher during distillation

    # --- Student Model Setup (already defined in _base_) ---
    # The student model configuration is inherited from bevformer_convnext_tiny.py

    # --- Distillation Loss Configuration ---
    distiller=dict(
        # This can be a generic distiller that iterates through losses
        type='FeatureLossDistiller',
        distill_losses=dict(
            # 1. Distill the final BEV feature map from BEVFormerEncoder output
            # This is a direct feature distillation (MSE) without attention guidance on the loss itself.
            loss_bev_encoder_output_map=dict(
                type='FeatureLoss',  # Using standard FeatureLoss
                # Example path: output of BEVFormerEncoder
                student_feature_loc='pts_bbox_head.transformer.encoder.saved_bev_features',
                teacher_feature_loc='pts_bbox_head.transformer.encoder.saved_bev_features',
                loss_cfg=dict(type='MSELoss', loss_weight=2.0,
                              reduction='mean'),
            ),

            # 2. Attention-guided distillation of TemporalSelfAttention output (last encoder layer)
            # Encoder has 3 layers (0, 1, 2). We use the last one (index 2).
            # attentions[0] in BEVFormerLayer is TemporalSelfAttention.
            loss_encoder_temporal_L2_guided=dict(
                type='AttentionGuidedFeatureLoss',  # Your custom loss
                student_feature_loc='pts_bbox_head.transformer.encoder.layers[2].attentions[0].saved_output_feature',
                teacher_feature_loc='pts_bbox_head.transformer.encoder.layers[2].attentions[0].saved_output_feature',
                teacher_attention_loc='pts_bbox_head.transformer.encoder.layers[2].attentions[0].saved_attention_map',
                base_loss_cfg=dict(type='MSELoss', reduction='mean'),
                loss_weight=1.0,  # Recommended weight
            ),

            # 3. Attention-guided distillation of SpatialCrossAttention output (last encoder layer)
            # attentions[1] in BEVFormerLayer is SpatialCrossAttention.
            # MSDeformableAttention3D is within SpatialCrossAttention.
            loss_encoder_spatial_L2_guided=dict(
                type='AttentionGuidedFeatureLoss',
                student_feature_loc='pts_bbox_head.transformer.encoder.layers[2].attentions[1].saved_output_feature',
                teacher_feature_loc='pts_bbox_head.transformer.encoder.layers[2].attentions[1].saved_output_feature',
                # Attention from MSDeformableAttention3D
                teacher_attention_loc='pts_bbox_head.transformer.encoder.layers[2].attentions[1].deformable_attention.saved_attention_map',
                base_loss_cfg=dict(type='MSELoss', reduction='mean'),
                loss_weight=1.5,  # Recommended weight
            ),

            # 4. Attention-guided distillation of Decoder's Cross-Attention output (last decoder layer)
            # Decoder has 6 layers (0-5). We use the last one (index 5).
            # attentions[1] in DetrTransformerDecoderLayer is CustomMSDeformableAttention (cross-attention).
            loss_decoder_cross_attn_L5_guided=dict(
                type='AttentionGuidedFeatureLoss',
                student_feature_loc='pts_bbox_head.transformer.decoder.layers[5].attentions[1].saved_output_feature',
                teacher_feature_loc='pts_bbox_head.transformer.decoder.layers[5].attentions[1].saved_output_feature',
                teacher_attention_loc='pts_bbox_head.transformer.decoder.layers[5].attentions[1].saved_attention_map',
                base_loss_cfg=dict(type='MSELoss', reduction='mean'),
                loss_weight=1.5,  # Recommended weight
            ),
        )
    )
)

# --- Optional: Adjust Training Strategy ---
# Distillation might benefit from longer training or different learning rates.
# You can override parts of the schedule or optimizer from the base config if needed.
# Example:
runner = dict(type='EpochBasedRunner', max_epochs=1000)  # Increase epochs
# lr_config = dict(step=[24, 33]) # Adjust LR schedule

# Ensure total_epochs matches runner.max_epochs for BEVFormer's custom training loop
total_epochs = 1000

# --- Custom Imports (if necessary) ---
# Point to the newly created modules
custom_imports = dict(
    imports=[
        'projects.bevformer_mods.distillers',  # If you have custom distillers
        'projects.bevformer_mods.losses',     # For AttentionGuidedFeatureLoss
        # Add paths to modified BEVFormer modules if they are in custom locations
        # e.g., 'projects.bevformer_mods.bevformer_modules'
    ],
    allow_failed_imports=False
)
