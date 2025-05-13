# projects/configs/bevformer/bevformer_convnext_tiny_distill_feature.py

# Inherit from the non-distilled ConvNeXt configuration
_base_ = ['./bevformer_convnext_tiny.py']

# --- Teacher Model Configuration ---
# Define the path to your pre-trained bevformer_tiny checkpoint
# !!! IMPORTANT: Replace this with the actual path to your checkpoint !!!
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
    type='DistillBEVFormer',  # Replace with your actual Distiller model type if different

    # --- Teacher Model Setup ---
    teacher_cfg=teacher_cfg_path,
    teacher_ckpt=teacher_ckpt,
    freeze_teacher=True,  # Usually freeze the teacher during distillation

    # --- Student Model Setup (already defined in _base_) ---
    # The student model configuration is inherited from bevformer_convnext_tiny.py

    # --- Distillation Loss Configuration ---
    distiller=dict(
        type='FeatureLossDistiller',  # A hypothetical distiller focusing on feature loss
        distill_losses=dict(
            # Define a name for this specific feature distillation loss
            loss_feature_neck=dict(
                type='FeatureLoss',  # A loss module comparing feature maps
                # Location to extract student features (output of img_neck)
                student_feature_loc='img_neck',
                # Location to extract teacher features (output of img_neck)
                teacher_feature_loc='img_neck',
                loss_cfg=dict(type='MSELoss', loss_weight=1.0,
                              reduction='mean'),  # Example: MSE Loss
                # --- Feature Dimension Adaptation (if needed) ---
                # If the channel dimensions of student and teacher features at 'img_neck'
                # do not match, you need an adapter layer.
                # Assuming bevformer_tiny neck output C=256 and convnext_tiny neck output C=256.
                # If they differ, uncomment and configure the adapter:
                # adapt_cfg=dict(
                #     type='Conv2dAdapter', # Example adapter type (e.g., a 1x1 Conv)
                #     in_channels=STUDENT_NECK_OUTPUT_CHANNELS, # e.g., 512 if neck doesn't reduce
                #     out_channels=TEACHER_NECK_OUTPUT_CHANNELS, # e.g., 256
                #     kernel_size=1
                # )
            )
            # You could potentially add more feature losses here, e.g., for BEV features:
            # loss_feature_bev=dict(
            #     type='FeatureLoss',
            #     student_feature_loc='pts_bbox_head.bev_encoder', # Example BEV feature location
            #     teacher_feature_loc='pts_bbox_head.bev_encoder', # Example BEV feature location
            #     loss_cfg=dict(type='MSELoss', loss_weight=2.0, reduction='mean'),
            # )
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
        'projects.bevformer_mods.distillers',
        'projects.bevformer_mods.losses'
    ],
    allow_failed_imports=False
)
