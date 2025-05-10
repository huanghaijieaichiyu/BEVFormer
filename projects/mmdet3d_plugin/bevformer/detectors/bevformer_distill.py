# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint, get_dist_info
from mmdet.models import build_detector
from mmdet3d.models.builder import DETECTORS
from mmdet.utils import get_root_logger
from .bevformer import BEVFormer
import copy

# Helper function to get BEV features from the encoder output


def get_encoder_bev_features(model_encoder_output, bev_h, bev_w, embed_dims):
    """
    Processes the output of BEVFormerEncoder to a standard (B, C, H, W) format.
    The actual output shape of BEVFormerEncoder (e.g., (bev_h*bev_w, bs, embed_dims))
    needs to be handled here.
    """
    # print(
    #     f"[Get BEV Debug] Input model_encoder_output type: {type(model_encoder_output)}")
    # if isinstance(model_encoder_output, torch.Tensor):
    #     print(
    #         f"[Get BEV Debug] Input model_encoder_output shape: {model_encoder_output.shape}, ndim: {model_encoder_output.ndim}")
    # print(
    #     f"[Get BEV Debug] Expected bev_h: {bev_h}, bev_w: {bev_w}, embed_dims: {embed_dims}, expected_shape[0]: {bev_h * bev_w}")

    # Handle teacher encoder output format: (bs, bev_h * bev_w, embed_dims)
    if isinstance(model_encoder_output, torch.Tensor) and \
       model_encoder_output.ndim == 3 and \
       model_encoder_output.shape[0] > 0 and \
       model_encoder_output.shape[1] == bev_h * bev_w and \
       model_encoder_output.shape[2] == embed_dims:
        bs = model_encoder_output.shape[0]
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        bev_features = model_encoder_output.permute(
            0, 2, 1).reshape(bs, embed_dims, bev_h, bev_w)
        # print(
        #     f"[Get BEV Debug] Processed (bs, H*W, C) format. Output shape: {bev_features.shape}")
        return bev_features

    # Handle original expected format: (bev_h*bev_w, bs, embed_dims)
    elif isinstance(model_encoder_output, torch.Tensor) and \
            model_encoder_output.ndim == 3 and \
            model_encoder_output.shape[0] == bev_h * bev_w and \
            model_encoder_output.shape[2] == embed_dims:  # Assuming shape[1] is bs
        bs = model_encoder_output.shape[1]
        # (H*W, B, C) -> (B, C, H*W) -> (B, C, H, W)
        bev_features = model_encoder_output.permute(
            1, 2, 0).reshape(bs, embed_dims, bev_h, bev_w)
        # print(
        #     f"[Get BEV Debug] Processed (H*W, B, C) format. Output shape: {bev_features.shape}")
        return bev_features

    # If it's already (B, C, H, W) or (B, embed_dims, H, W)
    elif isinstance(model_encoder_output, torch.Tensor) and model_encoder_output.ndim == 4:
        return model_encoder_output
    # If it's a tuple, e.g. (bev_embed, bev_pos) take the first element
    elif isinstance(model_encoder_output, tuple):
        # Recursive call on the first element, assuming it's the feature tensor
        return get_encoder_bev_features(model_encoder_output[0], bev_h, bev_w, embed_dims)
    else:
        # This part needs to be robust based on the exact output of your BEVFormerEncoder
        if isinstance(model_encoder_output, torch.Tensor):
            # Try to return if it's 4D and matches C, H, W somehow
            if model_encoder_output.ndim == 4 and model_encoder_output.shape[1] == embed_dims and \
               model_encoder_output.shape[2] == bev_h and model_encoder_output.shape[3] == bev_w:
                return model_encoder_output
        return None  # Or raise error


@DETECTORS.register_module()
class BEVFormerDistill(BEVFormer):
    def __init__(self,
                 teacher_cfg_path,  # Path to teacher's .py config file
                 teacher_checkpoint_path,  # Path to teacher's .pth checkpoint
                 distillation_cfg,
                 *args, **kwargs):  # Args and kwargs for student BEVFormer
        super().__init__(*args, **kwargs)  # Initialize student model parts

        self.distillation_cfg = distillation_cfg
        self.distill_loss_weight = self.distillation_cfg.get(
            'loss_weight', 1.0)
        self.distill_loss_type = self.distillation_cfg.get('loss_type', 'mse')

        # Load teacher model configuration
        teacher_mmcv_cfg = Config.fromfile(teacher_cfg_path)

        # Build teacher model
        logger = get_root_logger()
        logger.info(f"Building teacher model from {teacher_cfg_path}")

        # Ensure `train_cfg` and `test_cfg` for teacher are from its config or None if not training it
        self.teacher_model = build_detector(
            teacher_mmcv_cfg.model,  # Access the model dict from the loaded cfg
            train_cfg=teacher_mmcv_cfg.get('train_cfg'),  # Or None
            test_cfg=teacher_mmcv_cfg.get('test_cfg')  # Or None
        )

        # Load teacher checkpoint
        if teacher_checkpoint_path:
            logger.info(
                f"Loading teacher checkpoint from {teacher_checkpoint_path}")
            load_checkpoint(
                self.teacher_model, teacher_checkpoint_path, map_location='cpu', strict=True)
            logger.info(
                f"Teacher checkpoint loaded from {teacher_checkpoint_path}")

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Store teacher's BEV dimensions for feature processing
        self.teacher_bev_h = self.teacher_model.pts_bbox_head.bev_h
        self.teacher_bev_w = self.teacher_model.pts_bbox_head.bev_w
        self.teacher_embed_dims = self.teacher_model.pts_bbox_head.transformer.embed_dims

        # Store student's BEV dimensions for feature processing
        self.student_bev_h = self.pts_bbox_head.bev_h
        self.student_bev_w = self.pts_bbox_head.bev_w
        self.student_embed_dims = self.pts_bbox_head.transformer.embed_dims

        # Define adapter if BEV feature dimensions differ (primarily channel dimension)
        self.adapter = None
        if self.student_embed_dims != self.teacher_embed_dims:
            adapter_type = self.distillation_cfg.get('adapter', 'conv1x1')
            if adapter_type == 'conv1x1':
                self.adapter = nn.Conv2d(
                    self.student_embed_dims, self.teacher_embed_dims, kernel_size=1, bias=False)
                logger.info(
                    f"Initialized conv1x1 adapter from {self.student_embed_dims} to {self.teacher_embed_dims} channels.")
            elif adapter_type == 'linear':  # If features are (B, N, C)
                self.adapter = nn.Linear(
                    self.student_embed_dims, self.teacher_embed_dims, bias=False)
                logger.info(
                    f"Initialized linear adapter from {self.student_embed_dims} to {self.teacher_embed_dims} channels.")
            else:
                raise NotImplementedError(
                    f"Adapter type {adapter_type} not supported.")

    def _create_hook(self, storage_dict, model_output):
        # Process the encoder output to get (B, C, H, W) format
        # Use the dimensions of the model that produced this output (teacher)
        processed_bev = get_encoder_bev_features(
            model_output,
            self.teacher_bev_h,  # Use teacher's dimensions
            self.teacher_bev_w,
            self.teacher_embed_dims
        )
        storage_dict['value'] = processed_bev
        if processed_bev is None:
            print(
                f"Warning: _create_hook could not process model_output of type {type(model_output)} into expected BEV format.")

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None,
                      gt_labels=None, gt_bboxes=None, img=None, proposals=None,
                      gt_bboxes_ignore=None, **kwargs):

        if img is None:
            losses = dict()
            losses['loss_distill'] = torch.tensor(0.0, device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))  # Provide default device
            losses['loss_pts_cls'] = torch.tensor(0.0, device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
            losses['loss_pts_bbox'] = torch.tensor(0.0, device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
            print(
                "Warning: img is None in BEVFormerDistill.forward_train, returning zero losses.")
            return losses

        # Add similar check for img_metas
        if img_metas is None:
            losses = dict()
            # Use img's device/dtype if img is not None
            losses['loss_distill'] = torch.tensor(
                0.0, device=img.device, dtype=img.dtype)
            losses['loss_pts_cls'] = torch.tensor(
                0.0, device=img.device, dtype=img.dtype)
            losses['loss_pts_bbox'] = torch.tensor(
                0.0, device=img.device, dtype=img.dtype)
            print(
                "Warning: img_metas is None in BEVFormerDistill.forward_train, returning zero losses.")
            return losses

        # Standard BEVFormer preprocessing for student
        student_len_queue = img.size(1)  # Full queue length from input img
        student_prev_img = img[:, :-1, ...]  # History images for student BEV
        # Current frame image for student main path (5D)
        student_curr_img = img[:, -1, ...]

        # Deepcopy img_metas for teacher and student history processing to avoid in-place modifications
        teacher_img_metas_full_queue = copy.deepcopy(img_metas)
        student_history_img_metas = copy.deepcopy(img_metas)
        # Current frame metas for student (used for student's main bbox_head and some teacher calls)
        student_curr_img_metas = [each[student_len_queue-1]
                                  for each in img_metas]

        # --- Student: Obtain history BEV ---
        prev_bev = self.obtain_history_bev(
            student_prev_img, student_history_img_metas)
        if not student_curr_img_metas[0]['prev_bev_exists']:
            prev_bev = None  # Clear prev_bev if current frame is start of sequence

        # --- Student: Extract features for current frame ---
        student_extracted_features_tuple = self.extract_feat(
            img=student_curr_img,
            img_metas=student_curr_img_metas
        )
        # list[torch.Tensor]
        student_img_feats = student_extracted_features_tuple[0]

        # --- Teacher: Extract features for CURRENT frame (for distillation) ---
        # Teacher processes the same current frame as the student for feature comparison
        current_frame_img_for_teacher = student_curr_img  # Already 5D
        current_frame_img_metas_for_teacher = student_curr_img_metas

        # Dictionary to store teacher's BEV features from the encoder hook
        teacher_bev_features_from_hook = {'value': None}

        # Correct path to the teacher's BEV encoder for hook registration
        teacher_encoder_module = None
        if hasattr(self.teacher_model, 'pts_bbox_head') and self.teacher_model.pts_bbox_head and \
           hasattr(self.teacher_model.pts_bbox_head, 'transformer') and self.teacher_model.pts_bbox_head.transformer and \
           hasattr(self.teacher_model.pts_bbox_head.transformer, 'encoder') and self.teacher_model.pts_bbox_head.transformer.encoder:
            teacher_encoder_module = self.teacher_model.pts_bbox_head.transformer.encoder

        if teacher_encoder_module:
            hook_handle = teacher_encoder_module.register_forward_hook(
                lambda module, input, output: self._create_hook(
                    teacher_bev_features_from_hook, output)
            )
        else:
            hook_handle = None
            print("Warning: Teacher model BEV encoder (self.teacher_model.pts_bbox_head.transformer.encoder) not found for hook registration.")

        with torch.no_grad():
            # Teacher extracts features for the current frame.
            # len_queue=1 because teacher_curr_img is a single time slice.
            teacher_img_feats_tuple = self.teacher_model.extract_feat(
                img=current_frame_img_for_teacher,
                img_metas=current_frame_img_metas_for_teacher,
                len_queue=1
            )

            # Forward through teacher's pts_bbox_head to trigger encoder hook and get BEV features.
            # Teacher might need its own prev_bev if its transformer uses it. For simplicity, if not doing
            # complex temporal distillation with teacher, teacher_prev_bev can be None or handled separately.
            # Assuming for now teacher's encoder doesn't strictly need a prev_bev for just getting current features.
            if hasattr(self.teacher_model, 'pts_bbox_head') and self.teacher_model.pts_bbox_head:
                _ = self.teacher_model.pts_bbox_head(
                    teacher_img_feats_tuple[0],
                    current_frame_img_metas_for_teacher,
                    prev_bev=None,  # Or a teacher-specific prev_bev if its architecture requires it
                    only_bev=True  # To get BEV features without full head processing if not needed
                )
            else:
                print(
                    "Warning: Teacher model pts_bbox_head not found for BEV feature extraction.")

        if hook_handle:  # Remove hook only if it was registered
            hook_handle.remove()
        distillation_student_bev = prev_bev  # Student's BEV just after its encoder
        distillation_teacher_bev = teacher_bev_features_from_hook['value']

        # --- Calculate Distillation Loss ---
        if distillation_teacher_bev is not None and distillation_student_bev is not None:
            # Adapter for channel mismatch if necessary
            # Check last dim for C
            if self.distillation_cfg.get('adapter') and hasattr(self, 'adapter') and self.adapter is not None and distillation_student_bev.shape[-1] == self.student_embed_dims:

                student_bev_for_adapter = distillation_student_bev
                # Reshape student BEV if adapter is Conv2d and input is 3D (B, H*W, C)
                if isinstance(self.adapter, nn.Conv2d) and student_bev_for_adapter.ndim == 3:
                    # Expected B, H*W, C -> B, C, H, W
                    # H = self.student_bev_h, W = self.student_bev_w, C = self.student_embed_dims
                    # Batch size is student_bev_for_adapter.shape[0]
                    B = student_bev_for_adapter.shape[0]
                    # Check if H*W dimension matches expected H*W
                    if student_bev_for_adapter.shape[1] == self.student_bev_h * self.student_bev_w:
                        student_bev_for_adapter = student_bev_for_adapter.permute(0, 2, 1).view(
                            B, self.student_embed_dims, self.student_bev_h, self.student_bev_w)
                    else:
                        # This case would be an unexpected shape for student_bev_for_adapter
                        print(
                            f"Warning: Student BEV shape {student_bev_for_adapter.shape} unexpected for Conv2d adapter reshape.")

                # Now, student_bev_for_adapter should be in the correct shape for the adapter
                adapted_student_bev = self.adapter(student_bev_for_adapter)
            else:
                adapted_student_bev = distillation_student_bev

            # Ensure spatial dimensions match, or use an adapter/pooling for that too
            if adapted_student_bev.shape[2:] != distillation_teacher_bev.shape[2:]:
                # Simple interpolation if spatial sizes mismatch. More complex adapters could be used.
                distillation_teacher_bev_resized = F.interpolate(
                    distillation_teacher_bev, size=adapted_student_bev.shape[2:], mode='bilinear', align_corners=False
                )
            else:
                distillation_teacher_bev_resized = distillation_teacher_bev

            if self.distillation_cfg['loss_type'] == 'mse':
                loss_distill = F.mse_loss(
                    adapted_student_bev, distillation_teacher_bev_resized)
            elif self.distillation_cfg['loss_type'] == 'l1':
                loss_distill = F.l1_loss(
                    adapted_student_bev, distillation_teacher_bev_resized)
            else:
                raise ValueError(
                    f"Unsupported distillation loss_type: {self.distillation_cfg['loss_type']}")
            loss_distill = loss_distill * self.distillation_cfg['loss_weight']
        else:
            # Create a zero loss if BEV features are not available (e.g., during initial steps or if hook failed)
            # This ensures that 'loss_distill' is always present in the losses dictionary.
            device = student_curr_img.device if student_curr_img is not None else torch.device(
                "cpu")
            dtype = student_curr_img.dtype if student_curr_img is not None else torch.float32
            loss_distill = torch.tensor(0.0, device=device, dtype=dtype)

        # --- Student: bbox_head forward and loss calculation ---
        losses = dict()
        losses['loss_distill'] = loss_distill  # Add distillation loss

        # Student forward pass for detection losses
        # prev_bev here is the one obtained from student_prev_img
        student_outs = self.pts_bbox_head(
            student_img_feats, student_curr_img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, student_outs]
        student_detection_losses = self.pts_bbox_head.loss(
            *loss_inputs, img_metas=student_curr_img_metas)
        losses.update(student_detection_losses)

        return losses

    # Initialize adapter weights
    def init_weights(self):
        super().init_weights()
        if self.adapter is not None:
            if isinstance(self.adapter, nn.Conv2d) or isinstance(self.adapter, nn.Linear):
                # Example: Kaiming init for Conv2d, default for Linear is often fine
                if isinstance(self.adapter, nn.Conv2d):
                    nn.init.kaiming_uniform_(self.adapter.weight, a=1)
                if hasattr(self.adapter, 'bias') and self.adapter.bias is not None:
                    nn.init.constant_(self.adapter.bias, 0)
            logger = get_root_logger()
            logger.info("Adapter weights initialized.")
