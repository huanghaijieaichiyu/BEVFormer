# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images.
        Args:
            img (torch.Tensor): Input images of shape (B, N, C, H, W).
                                B is batch size, N is number of cameras or (QueueL * NumCameras).
            img_metas (list[dict]): Meta information of samples.
        Returns:
            list[torch.Tensor]: Image features after backbone and neck.
                                Each tensor in the list has shape (B, N, C_feat, H_feat, W_feat).
        """
        if img is None:
            return None

        B, N, C, H, W = img.shape
        # Reshape to (B*N, C, H, W) for backbone
        img_reshaped = img.view(B * N, C, H, W)

        if self.use_grid_mask:
            # # --- DEBUG PRINT FOR GRIDMASK ---
            # print(f"[BEVFormer GridMask Debug] img_reshaped.shape before grid_mask: {img_reshaped.shape}, self.use_grid_mask: {self.use_grid_mask}")
            # # --- END DEBUG PRINT ---
            img_reshaped = self.grid_mask(img_reshaped)

        # Pass the (potentially masked) 4D tensor to the image backbone
        x = self.img_backbone(img_reshaped)

        if self.with_img_neck:
            # x is typically a list of feature maps from FPN e.g. [(B*N, C1, H1, W1), ...]
            x = self.img_neck(x)

        # Reshape features back to include NumViews/Queue dimension
        img_feats_output = []
        if isinstance(x, (list, tuple)):  # Handle FPN outputs (list of tensors)
            for feat_map in x:
                _, C_feat, H_feat, W_feat = feat_map.shape
                # Reshape to (B, N, C_feat, H_feat, W_feat)
                img_feats_output.append(
                    feat_map.view(B, N, C_feat, H_feat, W_feat))
        elif isinstance(x, torch.Tensor):  # Handle single tensor output from neck/backbone
            _, C_feat, H_feat, W_feat = x.shape
            img_feats_output.append(x.view(B, N, C_feat, H_feat, W_feat))
        else:
            # Should not happen with standard backbones/necks
            raise TypeError(f"Unsupported type for FPN/Neck output: {type(x)}")

        return img_feats_output

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        # print(f"BEVFormer.extract_feat - Input img shape: {img.shape}") # Debug GridMask issue
        # The error happens in teacher model's grid_mask call. So, if this is teacher, img is 5D.
        # If use_grid_mask is True for the teacher, it needs to handle 5D or apply to 4D slices.

        # Let's check if grid_mask is active and how it's called in the teacher.
        # The traceback points to `teacher_model.extract_img_feat` then `teacher_model.grid_mask(img)`
        # In `extract_img_feat`, `img` is reshaped to 4D *before* backbone, but `grid_mask` is applied *after* neck.
        # The variable `img` passed to `grid_mask` in `extract_img_feat` (line 83 of bevformer.py in traceback)
        # is *NOT* the direct 5D input to `extract_img_feat`. It's the output from `img_neck`.
        # The error `n,c,h,w = x.size()` implies `x` (which is `img` in that context) is not 4D.

        # The original code in extract_img_feat is:
        # if self.use_grid_mask:
        #    img = self.grid_mask(img) <--- THIS `img` is the one causing issues if it's not 4D.
        # This `img` is the *output* of the neck (variable `x` in my modified snippet above).

        # The most direct way to check is to see the shape of the tensor passed to grid_mask
        # within the `extract_img_feat` of the *teacher* model.
        # The current `extract_img_feat` in `bevformer.py` applies grid_mask on the *output of the neck* (`x`).
        # So, if `x` from the neck is not 4D, it will fail.

        # Let's adjust the `extract_img_feat` in the student model (bevformer.py) to ensure the argument to grid_mask is 4D.
        # The provided traceback shows the error in the *teacher* model's call to grid_mask.
        # When `self.teacher_model.extract_feat` is called, it internally calls `self.teacher_model.extract_img_feat`.
        # Inside the *teacher's* `extract_img_feat` (which is the standard BEVFormer.extract_img_feat):
        #   `img` (5D) -> reshaped to 4D -> backbone -> neck -> output `x` (list of 4D tensors or single 4D tensor)
        #   Then, if `use_grid_mask` is true for the teacher:
        #       if `x` is a list (multiple FPN outputs): iterates `feat` in `x`, calls `grid_mask(feat)` -> feat should be 4D
        #       if `x` is a tensor (single FPN output): calls `grid_mask(x)` -> x should be 4D

        # The error `ValueError: too many values to unpack (expected 4)` suggests that the tensor
        # passed to `grid_mask` (i.e., `feat` or `x` after the neck) is NOT 4D in the teacher model's execution path.
        # This is unexpected if the neck always outputs 4D tensors.

        # One possibility: The teacher model's `img_neck` might be configured differently or behaving differently.
        # Let's ensure the `grid_mask` is only applied to 4D tensors within `extract_img_feat`.

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        pts_feats = None  # BEVFormer does not use LiDAR
        return (img_feats, pts_feats)

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)

            # extract_feat returns a tuple (img_feats, pts_feats)
            # We only need the img_feats part which is the first element of the tuple.
            extracted_features_tuple = self.extract_feat(
                img=imgs_queue, len_queue=len_queue)
            actual_img_feats_list = extracted_features_tuple[0]

            # Reshape features to recover time dimension
            # Each feat_map is currently (bs*len_queue, num_cams, C, H, W)
            # We need to reshape it to (bs, len_queue, num_cams, C, H, W)
            reshaped_img_feats = []
            for feat_map in actual_img_feats_list:
                B_L, num_cam, C_f, H_f, W_f = feat_map.shape
                reshaped = feat_map.view(bs, len_queue, num_cam, C_f, H_f, W_f)
                reshaped_img_feats.append(reshaped)

            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None

                # Now select the i-th time frame from each feature map
                # This will preserve the camera dimension and give us (bs, num_cams, C, H, W)
                img_feats_for_this_frame = [each_scale[:, i]
                                            for each_scale in reshaped_img_feats]

                prev_bev = self.pts_bbox_head(
                    img_feats_for_this_frame, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        # Correctly extract the image features list from the tuple returned by extract_feat
        extracted_features_tuple = self.extract_feat(
            img=img, img_metas=img_metas)
        # img_feats is now list[torch.Tensor]
        img_feats = extracted_features_tuple[0]

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function of point cloud branch."""
        # Ensure that x (mlvl_feats for the head) is the list of tensors, not the tuple from extract_feat
        # If x is the direct output of extract_feat, it's a tuple (img_feats_list, pts_feats_list)
        # We need img_feats_list for pts_bbox_head
        if isinstance(x, tuple) and len(x) == 2:
            img_feats_list = x[0]  # This should be list[torch.Tensor]
        # If x is already the list of features (e.g. passed from somewhere else)
        elif isinstance(x, list):
            img_feats_list = x
        else:
            # This case should ideally not be reached if inputs are consistent
            raise ValueError(
                f"Unexpected type for 'x' in simple_test_pts: {type(x)}. Expected tuple or list.")

        outs = self.pts_bbox_head(img_feats_list, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
