import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_loss, LOSSES
from mmcv.cnn import build_activation_layer, build_norm_layer
import warnings
from mmdet.models.builder import LOSSES as MMDetection_LOSSES
from mmdet.models.losses.utils import weighted_loss
from functools import partial

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
        loss_func (dict): Configuration for the core loss function (e.g., MSELoss).
        channel_adapter (dict, optional): Configuration for the adapter to match
                                          student channel dims to teacher's.
        spatial_adapter (dict, optional): Configuration for the adapter to match
                                          teacher spatial dims to student's.
    """

    def __init__(self,
                 student_feature_loc,
                 teacher_feature_loc,
                 loss_func,
                 channel_adapter=None,
                 spatial_adapter=None):
        super().__init__()
        self.student_feature_loc = student_feature_loc
        self.teacher_feature_loc = teacher_feature_loc

        # Build the core loss function (e.g., MSELoss)
        self.loss_func = build_loss(loss_func)

        # Build adapters
        self.channel_adapter = self._build_adapter(
            channel_adapter) if channel_adapter else None
        self.spatial_adapter = self._build_adapter(
            spatial_adapter) if spatial_adapter else None

    def _build_adapter(self, cfg):
        """Builds an adapter layer from a configuration dict."""
        adapter_type = cfg.get('type')
        if adapter_type == 'Conv2dAdapter':
            return nn.Conv2d(
                in_channels=cfg['in_channels'],
                out_channels=cfg['out_channels'],
                kernel_size=cfg.get('kernel_size', 1),
            )
        elif adapter_type == 'Conv1dAdapter':
            return nn.Conv1d(
                in_channels=cfg['in_channels'],
                out_channels=cfg['out_channels'],
                kernel_size=cfg.get('kernel_size', 1),
            )
        elif adapter_type == 'BilinearInterpolation':
            return partial(F.interpolate,
                           size=cfg['target_size'],
                           mode='bilinear',
                           align_corners=False)
        elif adapter_type is None:
            return None
        else:
            raise NotImplementedError(
                f"Adapter type '{adapter_type}' not implemented.")

    def forward(self, student_feature, teacher_feature):
        """
        Computes the feature distillation loss.
        Handles 3D (B, N, C) BEV features and 4D (B, C, H, W) image features.
        """
        # --- Select the primary feature tensor if input is a list/tuple ---
        s_feat = student_feature[0] if isinstance(
            student_feature, (list, tuple)) else student_feature
        t_feat = teacher_feature[0] if isinstance(
            teacher_feature, (list, tuple)) else teacher_feature

        # --- Spatial Adaptation (Teacher -> Student) ---
        if self.spatial_adapter:
            # Assumes 4D image feature (B, C, H, W) for interpolation
            if t_feat.ndim == 4:
                t_feat = self.spatial_adapter(t_feat)
            else:
                warnings.warn(
                    f"Spatial adapter not applied for teacher feature with dim={t_feat.ndim}")

        # --- Channel Adaptation (Student -> Teacher) ---
        if self.channel_adapter:
            # Assumes 3D BEV (B,N,C) needs permute or 4D Image (B,C,H,W)
            if s_feat.ndim == 3:  # (B, N, C) -> (B, C, N) for Conv1d
                s_feat = s_feat.permute(0, 2, 1)
                s_feat = self.channel_adapter(s_feat)
                s_feat = s_feat.permute(0, 2, 1)  # Back to (B, N, C)
            elif s_feat.ndim == 4:  # (B, C, H, W) for Conv2d
                s_feat = self.channel_adapter(s_feat)
            else:
                warnings.warn(
                    f"Channel adapter not applied for student feature with dim={s_feat.ndim}")

        # Final shape check
        if s_feat.shape != t_feat.shape:
            raise ValueError(
                f"Feature shape mismatch after adaptation! "
                f"Student: {s_feat.shape}, Teacher: {t_feat.shape}"
            )

        # Compute the loss
        loss = self.loss_func(s_feat, t_feat)
        return loss


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
