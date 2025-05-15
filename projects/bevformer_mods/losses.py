import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_loss, LOSSES
from mmcv.cnn import build_activation_layer, build_norm_layer
import warnings
from mmdet.models.builder import LOSSES as MMDetection_LOSSES
from mmdet.models.losses.utils import weighted_loss

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

# Helper function to recursively get an attribute from a nested object
def get_attr_recursive(obj, attr_string):
    parts = attr_string.split('.')
    for part in parts:
        obj = getattr(obj, part)
    return obj


@weighted_loss
def attention_guided_mse_loss(pred, target, attention_map):
    """
    Args:
        pred (torch.Tensor): The prediction. (Student Feature * Attention Map)
        target (torch.Tensor): The learning target. (Teacher Feature * Attention Map)
                             Or, if using (S-T)*A, target would be zeros.
        attention_map (torch.Tensor): The attention map from the teacher.
                                      (This might not be directly used here if already applied)
    """
    # Assuming attention is already applied to pred and target before passing here,
    # or that the loss is ( (S-T) * A_T )^2
    # If attention_map needs to be applied here:
    # diff = (pred - target) * attention_map
    # loss = F.mse_loss(diff, torch.zeros_like(diff), reduction='none')
    loss = F.mse_loss(pred, target, reduction='none')
    return loss


@LOSSES.register_module()
class AttentionGuidedFeatureLoss(nn.Module):
    """Attention Guided Feature Loss.

    Args:
        student_feature_loc (str): Dot-separated path to student's feature.
        teacher_feature_loc (str): Dot-separated path to teacher's feature.
        teacher_attention_loc (str): Dot-separated path to teacher's attention map.
        base_loss_cfg (dict): Configuration for the base loss function (e.g., MSELoss).
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        attention_norm_cfg (dict, optional): Configuration for normalizing attention maps.
            Example: dict(type='spatial_softmax') or dict(type='channel_norm', p=1)
            Defaults to None (no explicit normalization here, assumed done in model or before).
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean", "sum". Defaults to "mean".
    """

    def __init__(self,
                 student_feature_loc,
                 teacher_feature_loc,
                 teacher_attention_loc,
                 base_loss_cfg=dict(type='MSELoss', reduction='mean'),
                 loss_weight=1.0,
                 attention_norm_cfg=None,
                 reduction='mean',
                 **kwargs):
        super(AttentionGuidedFeatureLoss, self).__init__()
        self.student_feature_loc = student_feature_loc
        self.teacher_feature_loc = teacher_feature_loc
        self.teacher_attention_loc = teacher_attention_loc

        # Instantiate the base loss function
        # We need to handle how base_loss_cfg is used.
        # If base_loss_cfg is just for MSE, L1, etc., we can simplify.
        # For now, let's assume we'll use a custom weighted loss calculation or apply MSE directly.
        if base_loss_cfg['type'] == 'MSELoss':
            self.base_loss_func = F.mse_loss
        elif base_loss_cfg['type'] == 'L1Loss':
            self.base_loss_func = F.l1_loss
        else:
            raise NotImplementedError(
                f"Base loss type {base_loss_cfg['type']} not implemented.")

        self.loss_weight = loss_weight
        self.reduction = base_loss_cfg.get('reduction', reduction)
        self.attention_norm_cfg = attention_norm_cfg

    def _process_attention(self, attention_map, target_feature):
        """
        Process attention map:
        1. Ensure it's positive (e.g., after softmax if it's raw logits).
        2. Normalize it (e.g., sum to 1 over spatial/channel dims).
        3. Resize it to match the feature map if necessary.
        """
        # Example: Simple normalization (sum to 1 over spatial dimensions)
        # This is highly dependent on the attention map's original form and meaning.
        # For multi-head attention, you might average or max over heads.
        # B, NumHeads, H_attn, W_attn or B, QueryLen, KeyLen

        # Placeholder: Assume attention_map is already processed to some extent (e.g., averaged over heads)
        # and has spatial dimensions if applicable.
        # For BEV features (B, C, H, W), attention might be (B, 1, H, W) or (B, C, H, W)

        if attention_map.dim() > target_feature.dim() and attention_map.shape[1] > 1:
            attention_map = torch.mean(
                attention_map, dim=1, keepdim=True)  # Average over heads

        # Resize attention map to match feature map size if needed
        if attention_map.shape[-2:] != target_feature.shape[-2:]:
            attention_map = F.interpolate(
                attention_map, size=target_feature.shape[-2:], mode='bilinear', align_corners=False)

        # Normalize attention map (example: spatial min-max normalization per batch/channel)
        # This step is crucial and highly dependent on the nature of your attention maps.
        # A common approach is to ensure values are in [0, 1] or sum to 1.
        # For simplicity, let's do a min-max scaling if it's not already normalized.
        # This part needs careful consideration based on your specific attention maps.
        min_vals = attention_map.amin(dim=(-2, -1), keepdim=True)
        max_vals = attention_map.amax(dim=(-2, -1), keepdim=True)
        attention_map_norm = (attention_map - min_vals) / \
            (max_vals - min_vals + 1e-6)

        return attention_map_norm

    def forward(self, student_model, teacher_model, **kwargs):
        """
        Args:
            student_model (nn.Module): The student model.
            teacher_model (nn.Module): The teacher model.
        Returns:
            torch.Tensor: The calculated loss.
        """
        student_feat = get_attr_recursive(
            student_model, self.student_feature_loc)
        teacher_feat = get_attr_recursive(
            teacher_model, self.teacher_feature_loc)
        teacher_attn = get_attr_recursive(
            teacher_model, self.teacher_attention_loc)

        # Process teacher attention map
        # This is a critical step and might need customization.
        # For example, if attention maps are from multi-head attention,
        # you might average over heads, or select specific heads.
        # Ensure the attention map is broadcastable to the feature maps.
        processed_teacher_attn = self._process_attention(
            teacher_attn, teacher_feat)

        # Formulation: L = || A_T * (F_S - F_T) ||^2_2
        # Or: L = || A_T * F_S - A_T * F_T ||^2_2
        # Let's use the first one for clarity of weighting the difference.

        # Ensure features are detached if they are not meant to propagate gradients
        # Teacher features should always be detached.
        teacher_feat = teacher_feat.detach()
        processed_teacher_attn = processed_teacher_attn.detach()

        # Calculate the attention-weighted difference
        # The shapes must be compatible for element-wise multiplication.
        # If student_feat is (B, C, H, W) and processed_teacher_attn is (B, 1, H, W),
        # broadcasting will handle it. If processed_teacher_attn is (B, C, H, W), it's a direct match.

        # Make sure student and teacher features have the same shape
        if student_feat.shape != teacher_feat.shape:
            # This could happen if, for instance, student has different channel depth
            # This needs a strategy: e.g. an adapter, or ensure models are compatible for this loss.
            # For now, we'll raise an error or try a simple interpolation if only spatial dims differ.
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                student_feat = F.interpolate(
                    student_feat, size=teacher_feat.shape[2:], mode='bilinear', align_corners=False)
            # If channel dimensions differ, this loss as-is won't work without an adapter.
            if student_feat.shape[1] != teacher_feat.shape[1]:
                # TODO: Implement or require an adapter for channel mismatch
                print(
                    f"Warning: Channel mismatch for {self.student_feature_loc}. Student: {student_feat.shape[1]}, Teacher: {teacher_feat.shape[1]}. Loss may be incorrect.")

        # Original formulation from many papers: Guide student to match teacher's features
        # where teacher pays attention.
        # Loss = MSE( Student_Feature * Attention_Teacher, Teacher_Feature * Attention_Teacher)

        # Alternative: Focus on difference in attended regions
        # Loss = MSE ( (Student_Feature - Teacher_Feature) * Attention_Teacher, 0 )
        # This requires the base_loss_func to handle (prediction, zero_target)

        # Let's try: L = || A_T * F_S - A_T * F_T ||
        student_feat_weighted = student_feat * processed_teacher_attn
        teacher_feat_weighted = teacher_feat * processed_teacher_attn

        loss = self.base_loss_func(
            student_feat_weighted, teacher_feat_weighted, reduction='none')

        # Weighted sum of the loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # 'none' will be handled by loss_weight application

        return loss * self.loss_weight
