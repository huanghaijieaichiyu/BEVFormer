import torch
import torch.nn as nn
import timm
from mmcv.runner import BaseModule
from mmcv.utils import get_logger
from mmdet.models.builder import BACKBONES

@BACKBONES.register_module()
class SwinTimm(BaseModule):
    """Swin-Tiny backbone implemented via timm."""
    def __init__(self,
                 model_name='swin_tiny_patch4_window7_224',
                 pretrained=True,
                 out_indices=(3,),
                 drop_path_rate=0.1,
                 init_cfg=None,
                 norm_eval=True):
        # Use init_cfg only when pretrained=False
        effective_init_cfg = init_cfg if not pretrained else None
        super().__init__(init_cfg=effective_init_cfg)
        self.norm_eval = norm_eval
        if not isinstance(out_indices, (list, tuple)):
            out_indices = [out_indices]
        # Create timm model and use forward_features for backbone
        self.timm_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate
        )
        # Set output channels from timm model's num_features (final feature dim)
        self._out_channels = [self.timm_model.num_features]

    def forward(self, x):
        # Dynamically adjust patch_embed size and resolution to input resolution
        if hasattr(self.timm_model, 'patch_embed'):
            _, _, H, W = x.shape
            pe = self.timm_model.patch_embed
            # update image size
            pe.img_size = (H, W)
            # recompute patch grid resolution and total patches
            p = pe.patch_size
            pH, pW = (p, p) if isinstance(p, int) else p
            new_res = (H // pH, W // pW)
            # timm v0.7+ uses patches_resolution, older use grid_size
            if hasattr(pe, 'patches_resolution'):
                pe.patches_resolution = new_res
            if hasattr(pe, 'grid_size'):
                pe.grid_size = new_res
            pe.num_patches = new_res[0] * new_res[1]
            # update input_resolution for all layers and their blocks/downsample
            for i, layer in enumerate(self.timm_model.layers):
                # resolution of this stage
                res_i = (new_res[0] // (2 ** i), new_res[1] // (2 ** i))
                # update BasicLayer input_resolution
                setattr(layer, 'input_resolution', res_i)
                # update each SwinTransformerBlock
                if hasattr(layer, 'blocks'):
                    for block in layer.blocks:
                        setattr(block, 'input_resolution', res_i)
                # update PatchMerging in downsample
                if hasattr(layer, 'downsample') and layer.downsample is not None:
                    setattr(layer.downsample, 'input_resolution', res_i)
        # Extract features without classification head
        feat = self.timm_model.forward_features(x)
        return (feat,)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    @property
    def out_channels(self):
        return self._out_channels 