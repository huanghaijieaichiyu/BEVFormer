import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_loss, LOSSES
from mmcv.cnn import build_activation_layer, build_norm_layer
import warnings

# Builder function for distillation losses


def build_distill_loss(cfg):
    """Builds a distillation loss module from a configuration dict."""
    return LOSSES.build(cfg)


@LOSSES.register_module()
class FeatureLoss(nn.Module):
    """PyTorch module for computing feature-based distillation loss.

    Args:
        student_feature_loc (str): Location string for student feature extraction.
        teacher_feature_loc (str): Location string for teacher feature extraction.
        loss_cfg (dict): Configuration for the core loss function (e.g., MSELoss,
                         L1Loss). Must include 'type' and 'loss_weight'.
        adapt_cfg (dict, optional): Configuration for the adapter layer to match
                                    student features to teacher features. Defaults to None.
    """

    def __init__(self,
                 student_feature_loc,
                 teacher_feature_loc,
                 loss_cfg,
                 adapt_cfg=None):
        super().__init__()
        self.student_feature_loc = student_feature_loc
        self.teacher_feature_loc = teacher_feature_loc

        # Build the core loss function (e.g., MSELoss)
        if 'loss_weight' not in loss_cfg:
            warnings.warn(
                "'loss_weight' not specified in loss_cfg for FeatureLoss, defaulting to 1.0")
        self.loss_weight = loss_cfg.get('loss_weight', 1.0)
        # Keep the original weight in loss_cfg for the build_loss function
        # but use self.loss_weight for the final scaling
        # loss_cfg_copy = loss_cfg.copy()
        # loss_cfg_copy['loss_weight'] = 1.0 # Let build_loss use weight 1, we scale later
        self.loss_func = build_loss(loss_cfg)

        # Build the adapter layer if specified
        self.adapter = None
        if adapt_cfg:
            self.adapter = self._build_adapter(adapt_cfg)
            print(
                f"Built adapter for FeatureLoss: {adapt_cfg.get('type', 'N/A')}")
        else:
            print("No adapter configured for FeatureLoss.")

    def _build_adapter(self, adapt_cfg):
        """Builds the adapter layer based on config."""
        adapter_type = adapt_cfg.get('type')
        if adapter_type == 'Conv2dAdapter':
            in_channels = adapt_cfg.get('in_channels')
            out_channels = adapt_cfg.get('out_channels')
            kernel_size = adapt_cfg.get('kernel_size', 1)
            stride = adapt_cfg.get('stride', 1)
            padding = adapt_cfg.get('padding', 0)
            bias = adapt_cfg.get('bias', True)
            # Add checks for required args
            if not all([in_channels, out_channels]):
                raise ValueError(
                    "Conv2dAdapter requires 'in_channels' and 'out_channels' in adapt_cfg")
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=bias)
        elif adapter_type == 'LinearAdapter':  # Example for flattened features
            in_features = adapt_cfg.get('in_features')
            out_features = adapt_cfg.get('out_features')
            bias = adapt_cfg.get('bias', True)
            if not all([in_features, out_features]):
                raise ValueError(
                    "LinearAdapter requires 'in_features' and 'out_features' in adapt_cfg")
            return nn.Linear(in_features, out_features, bias=bias)
        # Add other adapter types if needed (e.g., MLP, Attention-based)
        else:
            raise NotImplementedError(
                f"Adapter type '{adapter_type}' not implemented.")

    def forward(self, student_feature, teacher_feature):
        """
        Computes the feature distillation loss.

        Args:
            student_feature (Tensor): Feature map from the student model.
            teacher_feature (Tensor): Feature map from the teacher model.

        Returns:
            torch.Tensor: Computed distillation loss, scaled by loss_weight.
        """
        # --- Handle potential list/tuple input from FPN ---
        # Select the first feature map (index 0) if input is a sequence
        if isinstance(student_feature, (list, tuple)):
            if not student_feature:
                warnings.warn(
                    "Student feature is an empty sequence. Skipping loss.")
                return torch.tensor(0.0, device=self.loss_func.loss_weight.device if hasattr(self.loss_func, 'loss_weight') else 'cpu', requires_grad=False)
            s_feat_selected = student_feature[0]
            # Ensure teacher feature is also a sequence and select the corresponding element
            if not isinstance(teacher_feature, (list, tuple)) or not teacher_feature:
                warnings.warn(
                    f"Student feature is a sequence, but teacher is not ({type(teacher_feature)}) or is empty. Skipping loss.")
                return torch.tensor(0.0, device=s_feat_selected.device, requires_grad=False)
            t_feat_selected = teacher_feature[0]
        elif isinstance(student_feature, torch.Tensor):
            # If student is a Tensor, teacher must also be a Tensor
            if not isinstance(teacher_feature, torch.Tensor):
                warnings.warn(
                    f"Student feature is a Tensor, but teacher is not ({type(teacher_feature)}). Skipping loss.")
                return torch.tensor(0.0, device=student_feature.device, requires_grad=False)
            s_feat_selected = student_feature
            t_feat_selected = teacher_feature
        else:
            # Handle unexpected types if necessary, or just let subsequent checks fail
            warnings.warn(
                f"Unexpected feature type received: {type(student_feature)}. Skipping loss.")
            # Attempt to find a device for the zero tensor
            device = 'cpu'
            if hasattr(self.loss_func, 'loss_weight') and isinstance(self.loss_func.loss_weight, torch.Tensor):
                device = self.loss_func.loss_weight.device
            return torch.tensor(0.0, device=device, requires_grad=False)

        # --- Proceed with the selected single tensors ---
        # Should not happen with checks above, but be safe
        if s_feat_selected is None or t_feat_selected is None:
            # Re-check device determination logic if needed
            device = 'cpu'
            if hasattr(self.loss_func, 'loss_weight') and isinstance(self.loss_func.loss_weight, torch.Tensor):
                device = self.loss_func.loss_weight.device
            return torch.tensor(0.0, device=device, requires_grad=False)

        # --- Feature Normalization (L2 norm along channel dimension) ---
        # Assuming features are (B, C, H, W) or (B, N, C, H, W) -> hook captures (B,N,C,H,W)
        # If features are (B, N, C, H, W), hooks will give this. Loss expects (B, C, H, W) or similar.
        # The FPN output selected s_feat_selected/t_feat_selected is (B*N, C, H, W)
        # or if single view, (B, C, H, W) based on BEVFormer.extract_img_feat logic.
        # Let's assume s_feat_selected and t_feat_selected are 4D: (Batch, Channels, Height, Width)

        if s_feat_selected.ndim == 4 and t_feat_selected.ndim == 4:
            # Normalize along channel dimension (dim=1)
            s_norm = F.normalize(s_feat_selected, p=2, dim=1)
            t_norm = F.normalize(t_feat_selected, p=2, dim=1)
        # (B, N, C, H, W)
        elif s_feat_selected.ndim == 5 and t_feat_selected.ndim == 5:
            # Reshape to (B*N, C, H, W), normalize, then reshape back if needed by loss or adapter
            # Or, normalize per (N, C, H, W) slice if that makes more sense. For now, flatten N.
            bs_s, n_s, c_s, h_s, w_s = s_feat_selected.shape
            bs_t, n_t, c_t, h_t, w_t = t_feat_selected.shape
            s_reshaped = s_feat_selected.view(bs_s * n_s, c_s, h_s, w_s)
            t_reshaped = t_feat_selected.view(bs_t * n_t, c_t, h_t, w_t)

            s_norm_reshaped = F.normalize(s_reshaped, p=2, dim=1)
            t_norm_reshaped = F.normalize(t_reshaped, p=2, dim=1)

            # For now, let's assume the adapter and loss expect 4D after this point
            # If the original 5D structure is crucial for adapter/loss, this needs adjustment.
            s_norm = s_norm_reshaped
            t_norm = t_norm_reshaped
            # Potential place to reshape back if adapter expects 5D:
            # s_norm = s_norm_reshaped.view(bs_s, n_s, c_s, h_s, w_s)
            # t_norm = t_norm_reshaped.view(bs_t, n_t, c_t, h_t, w_t)
        else:
            warnings.warn(
                f"Skipping normalization for features with unhandled dims: s_dim={s_feat_selected.ndim}, t_dim={t_feat_selected.ndim}")
            s_norm = s_feat_selected
            t_norm = t_feat_selected

        # Apply adapter to the selected & NORMALIZED student feature if configured
        if self.adapter:
            # Apply adapter to normalized feature
            adapted_student_feature = self.adapter(s_norm)
        else:
            adapted_student_feature = s_norm

        # Ensure selected & NORMALIZED features have compatible shapes (after potential adaptation)
        if adapted_student_feature.shape != t_norm.shape:
            warnings.warn(
                f"Selected feature shapes mismatch after adaptation/normalization! "
                f"Student: {adapted_student_feature.shape}, Teacher (normalized): {t_norm.shape}. "
                f"Attempting spatial resize if possible (usually indicates config error)."
            )
            if adapted_student_feature.ndim == 4 and t_norm.ndim == 4:
                s_h, s_w = adapted_student_feature.shape[-2:]
                t_h, t_w = t_norm.shape[-2:]
                if (s_h, s_w) != (t_h, t_w):
                    adapted_student_feature = F.interpolate(adapted_student_feature,
                                                            size=(t_h, t_w),
                                                            mode='bilinear',
                                                            align_corners=False)
                if adapted_student_feature.shape[1] != t_norm.shape[1]:
                    raise ValueError(f"Channel mismatch persists after spatial resize: "
                                     f"{adapted_student_feature.shape[1]} vs {t_norm.shape[1]}")

            elif adapted_student_feature.shape[0] != t_norm.shape[0] or \
                    adapted_student_feature.shape[1] != t_norm.shape[1]:
                raise ValueError(
                    f"Feature shapes mismatch (non-spatial or non-4D)! "
                    f"Student: {adapted_student_feature.shape}, Teacher (normalized): {t_norm.shape}. "
                    f"Adapter might be misconfigured or missing."
                )
            if adapted_student_feature.shape != t_norm.shape:
                raise ValueError(
                    f"Could not resolve feature shape mismatch! "
                    f"Student (adapted): {adapted_student_feature.shape}, Teacher (normalized): {t_norm.shape}. "
                )

        # Compute the loss using the selected, NORMALIZED & potentially adapted features
        loss = self.loss_func(adapted_student_feature, t_norm)

        # Scale the loss by the specified weight
        return loss * self.loss_weight


# Optional: Define a simple Conv2dAdapter explicitly if needed outside FeatureLoss
# @MODELS.register_module() # Or just use nn.Conv2d directly in config
# class Conv2dAdapter(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
