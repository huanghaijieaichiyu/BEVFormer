import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from mmdet.models import build_detector, DETECTORS
from mmdet3d.models import builder
import warnings
from functools import partial

# Assuming FeatureLossDistiller might be defined elsewhere or handled within this class
# from .losses import build_distiller # Example if distiller logic is separate


def _load_checkpoint_teacher(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint file for teacher, ignoring unexpected keys."""
    checkpoint = _load_checkpoint(filename, map_location)
    # Get state_dict from checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Load state_dict
    load_state_dict(model, state_dict, strict, logger)
    warnings.warn("Teacher weights loaded and frozen.", UserWarning)
    return checkpoint


@DETECTORS.register_module()
class DistillBEVFormer(BEVFormer):
    """
    Distiller wrapper for BEVFormer models using Feature Map Distillation.
    Inherits from BEVFormer to handle student model initialization and methods.

    Args:
        teacher_cfg (str or dict): Config file path or dict for the teacher model.
        teacher_ckpt (str): Path to the teacher model checkpoint.
        distiller (dict): Configuration for the distillation losses and logic.
        freeze_teacher (bool): Whether to freeze the teacher model weights. Defaults to True.
        **kwargs: Arguments passed to the BEVFormer (student) init method
                 (e.g., img_backbone, pts_bbox_head, use_grid_mask, etc.).
    """

    def __init__(self,
                 teacher_cfg,
                 teacher_ckpt,
                 distiller,
                 freeze_teacher=True,
                 **kwargs):  # kwargs contains student config like use_grid_mask

        # 1. Initialize the Student Model (BEVFormer) first
        super().__init__(**kwargs)
        print(
            f"Student BEVFormer initialized with args: {list(kwargs.keys())}")

        # 2. Build Teacher Model
        if isinstance(teacher_cfg, str):
            from mmcv import Config
            teacher_cfg = Config.fromfile(teacher_cfg)
        teacher_cfg.model.pretrained = None
        if 'init_cfg' in teacher_cfg.model:
            teacher_cfg.model.init_cfg = None
        # Use the builder from mmdet3d which might be more appropriate
        self.teacher = builder.build_detector(teacher_cfg.model)

        # 3. Load Teacher Checkpoint
        print(f"Attempting to load teacher weights from: {teacher_ckpt}")
        if teacher_ckpt:
            try:
                _load_checkpoint_teacher(
                    self.teacher, teacher_ckpt, map_location='cpu', strict=False)
                print(
                    f"Teacher weights loaded successfully from: {teacher_ckpt}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load teacher checkpoint '{teacher_ckpt}': {e}. Teacher has random weights.")
        else:
            warnings.warn(
                "No teacher checkpoint provided! Teacher has random weights.", UserWarning)

        # 4. Freeze Teacher
        if freeze_teacher:
            self.freeze_teacher()

        # 5. Prepare Distiller Losses
        self.distiller_cfg = distiller
        self._build_distiller_losses()

        # 6. Setup Hooks for Feature Extraction
        self.student_feature_locs = set()
        self.teacher_feature_locs = set()
        if hasattr(self, 'distill_losses') and self.distill_losses:
            for loss_name, loss_module in self.distill_losses.items():
                if hasattr(loss_module, 'student_feature_loc'):
                    self.student_feature_locs.add(
                        loss_module.student_feature_loc)
                if hasattr(loss_module, 'teacher_feature_loc'):
                    self.teacher_feature_locs.add(
                        loss_module.teacher_feature_loc)
                    # Ensure teacher model also has the required feature location module
                    if self._find_module(self.teacher, loss_module.teacher_feature_loc) is None:
                        warnings.warn(
                            f"Distillation loss '{loss_name}' requires teacher feature '{loss_module.teacher_feature_loc}', but module not found in teacher model.")

        # Initialize hook-related attributes *before* registering
        self.student_features = {}
        self.teacher_features = {}
        self._student_hook_handles = []
        self._teacher_hook_handles = []
        self._register_hooks()  # Register hooks after student and teacher are built

        print("DistillBEVFormer fully initialized.")

    def _build_distiller_losses(self):
        """Build the loss modules specified in the distiller config."""
        self.distill_losses = nn.ModuleDict()
        if self.distiller_cfg and 'distill_losses' in self.distiller_cfg:
            for loss_name, loss_cfg in self.distiller_cfg['distill_losses'].items():
                # Assumes a FeatureLoss class or similar exists and is registered
                # Loss type (e.g., 'FeatureLoss') should be in loss_cfg
                # We might need a specific builder for distillation losses
                try:
                    # Use mmdet3d's builder if applicable, or a custom one
                    from projects.bevformer_mods.losses import build_distill_loss
                    self.distill_losses[loss_name] = build_distill_loss(
                        loss_cfg)
                    print(
                        f"Built distillation loss: {loss_name} ({loss_cfg.get('type', 'N/A')})")
                except ImportError:
                    warnings.warn(
                        "Could not import build_distill_loss. Ensure projects.bevformer_mods.losses exists.", UserWarning)
                    # Fallback or raise error
                    raise

    def _find_module(self, model, module_path):
        """Find a module in a model using a dot-separated path."""
        parts = module_path.split('.')
        module = model
        try:
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            warnings.warn(
                f"Could not find module at path: {module_path}", UserWarning)
            return None

    def _student_hook(self, module, input, output, key):
        """Hook function to capture student features."""
        self.student_features[key] = output

    def _teacher_hook(self, module, input, output, key):
        """Hook function to capture teacher features."""
        self.teacher_features[key] = output

    def _register_hooks(self):
        """Register forward hooks to capture features."""
        self._remove_hooks()  # Ensure clean state

        for loc in self.student_feature_locs:
            module = self._find_module(self, loc)  # Student is self
            if module is not None:
                handle = module.register_forward_hook(
                    partial(self._student_hook, key=loc))
                self._student_hook_handles.append(handle)
                print(f"Registered student hook for: {loc}")

        for loc in self.teacher_feature_locs:
            module = self._find_module(self.teacher, loc)
            if module is not None:
                handle = module.register_forward_hook(
                    partial(self._teacher_hook, key=loc))
                self._teacher_hook_handles.append(handle)
                print(f"Registered teacher hook for: {loc}")

        if not self._student_hook_handles and self.student_feature_locs:
            warnings.warn(
                "Failed to register any student hooks despite requests.")
        if not self._teacher_hook_handles and self.teacher_feature_locs:
            warnings.warn(
                "Failed to register any teacher hooks despite requests.")

    def _remove_hooks(self):
        """Remove all registered forward hooks."""
        for handle in self._student_hook_handles:
            handle.remove()
        for handle in self._teacher_hook_handles:
            handle.remove()
        self._student_hook_handles.clear()
        self._teacher_hook_handles.clear()

    def forward_train(self, **kwargs):
        """
        Forward pass for training with distillation.
        Args:
            kwargs: Input data (e.g., img, img_metas, gt_bboxes_3d, gt_labels_3d).
        """
        self.student_features.clear()
        self.teacher_features.clear()

        # Teacher forward pass (no_grad context is crucial)
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                _ = self.teacher(**kwargs)  # Hooks capture features
        else:
            warnings.warn(
                "Teacher model not available for forward pass during training.")

        # Student forward pass and base task loss calculation
        # super().forward_train calls the BEVFormer's forward_train
        student_task_losses = super().forward_train(**kwargs)
        if not isinstance(student_task_losses, dict):
            warnings.warn(
                f"Student forward_train returned non-dict type: {type(student_task_losses)}. Initializing losses as empty dict.")
            student_task_losses = {}

        # Calculate Distillation Losses
        losses_distill = dict()
        if hasattr(self, 'distill_losses') and self.distill_losses:
            for loss_name, loss_module in self.distill_losses.items():
                s_feat = self.student_features.get(
                    loss_module.student_feature_loc)
                t_feat = self.teacher_features.get(
                    loss_module.teacher_feature_loc)

                if s_feat is not None and t_feat is not None:
                    loss_val = loss_module(s_feat, t_feat)
                    losses_distill[loss_name] = loss_val
                else:
                    missing = []
                    if s_feat is None:
                        missing.append(
                            f"student ('{loss_module.student_feature_loc}')")
                    if t_feat is None:
                        missing.append(
                            f"teacher ('{loss_module.teacher_feature_loc}')")
                    warnings.warn(
                        f"Skipping distillation loss '{loss_name}' due to missing features: {', '.join(missing)}.")

        # Combine Losses
        student_task_losses.update(losses_distill)

        if not student_task_losses:
            warnings.warn(
                "No losses computed (neither task nor distillation). Returning empty dict.")

        return student_task_losses

    # Remove methods that should be inherited directly from BEVFormer
    # unless specific modification is needed for distillation (forward_test usually isn't)
    # def forward_test(self, **kwargs): ...
    # def extract_feat(self, **kwargs): ...
    # def simple_test(self, **kwargs): ...
    # def aug_test(self, **kwargs): ...

    # Keep the train method override to ensure teacher stays in eval mode
    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.eval()

    # Keep __del__ to remove hooks
    def __del__(self):
        # Check if hooks were initialized before trying to remove
        if hasattr(self, '_student_hook_handles') and hasattr(self, '_teacher_hook_handles'):
            self._remove_hooks()
            # print("DistillBEVFormer hooks removed.") # Optional print
        # else: # Debugging print if needed
            # print("DistillBEVFormer __del__ called, but hooks were not initialized.")
        pass  # Avoid errors if __init__ failed early

    def freeze_teacher(self):
        """Freeze teacher model parameters."""
        if hasattr(self, 'teacher') and self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
            print("Teacher model frozen.")
        else:
            warnings.warn("No teacher model to freeze.", UserWarning)
