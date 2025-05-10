import torch
import torch.nn as nn
import timm
from mmcv.runner import BaseModule, load_checkpoint
# from mmcv.cnn import build_norm_layer # Not strictly needed if timm handles norms
from mmdet.models.builder import BACKBONES
# from mmengine.logging import MMLogger # Use get_logger for older mmcv/mmdet
from mmcv.utils import get_logger


@BACKBONES.register_module()
class ConvNeXtTimm(BaseModule):
    """ConvNeXt backbone based on timm library.

    Args:
        model_name (str): Name of the ConvNeXt model in timm.
            Defaults to 'convnext_tiny'.
        pretrained (bool): Whether to load timm's pretrained weights.
            Defaults to True.
        out_indices (tuple[int]): Output from which stages.
            Defaults to (3,). For convnext_tiny, (0,1,2,3) correspond to
            features with channels (96, 192, 384, 768).
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        norm_eval (bool): Whether to set norm layers to eval mode,
            affecting only BatchNorm layers. ConvNeXt uses LayerNorm.
            Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Useful if pretrained=False and you want to load a custom checkpoint.
            Defaults to None.
    """

    def __init__(self,
                 model_name='convnext_tiny',
                 pretrained=True,
                 out_indices=(3,),
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6,
                 norm_eval=True,
                 # with_cp=False, # timm models might not directly support mmcv-style checkpointing
                 init_cfg=None):
        # Handle init_cfg priority:
        # 1. If init_cfg is for timm's pretrained, timm handles it.
        # 2. If init_cfg is 'Pretrained' and pretrained=False, we load it.
        # 3. If pretrained=True, timm handles loading, init_cfg for BaseModule can be None.
        effective_init_cfg = init_cfg
        if pretrained:
            # If timm is handling pretraining, BaseModule's init_cfg can be None
            # unless a custom one is forced (e.g. for fine-tuning specific parts later)
            effective_init_cfg = None

        super().__init__(init_cfg=effective_init_cfg)

        logger = get_logger('mmdet')  # Use MMDetection's logger

        self.norm_eval = norm_eval
        # self.with_cp = with_cp # Not directly used with timm models here

        if not isinstance(out_indices, (list, tuple)):
            out_indices = [out_indices]

        # Create timm model
        self.timm_model = timm.create_model(
            model_name,
            pretrained=pretrained,  # timm will handle download and load if True
            features_only=True,    # Crucial to get intermediate features
            out_indices=out_indices,
            drop_path_rate=drop_path_rate,
            ls_init_value=layer_scale_init_value,
        )

        # If pretrained is False and init_cfg is provided for a custom checkpoint
        if not pretrained and init_cfg and init_cfg.get('type') == 'Pretrained' and init_cfg.get('checkpoint'):
            checkpoint_path = init_cfg['checkpoint']
            logger.info(
                f"Loading custom pretrained weights for {model_name} from {checkpoint_path}")
            # For timm models, direct state_dict loading is often cleaner if not using mmcv's _load_checkpoint nuances
            # However, _load_checkpoint can handle prefix mapping if needed.
            # We'll try a simpler load first, assuming the checkpoint is compatible or from timm.
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                # Timm models often have a 'model' or 'state_dict' key in checkpoints from other frameworks
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

                # Handle potential "module." prefix from DataParallel/DistributedDataParallel
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v

                missing_keys, unexpected_keys = self.timm_model.load_state_dict(
                    new_state_dict, strict=False)
                if missing_keys:
                    logger.warning(
                        f"Missing keys when loading checkpoint for {model_name}: {missing_keys}")
                if unexpected_keys:
                    logger.warning(
                        f"Unexpected keys when loading checkpoint for {model_name}: {unexpected_keys}")
                logger.info(
                    f"Successfully loaded custom weights for {model_name} from {checkpoint_path}")

            except Exception as e:
                logger.error(
                    f"Failed to load custom checkpoint {checkpoint_path} for {model_name}: {e}")
                logger.info(
                    "Falling back to mmcv.runner.load_checkpoint for more robust loading")
                load_checkpoint(self.timm_model, checkpoint_path,
                                map_location='cpu', strict=False, logger=logger)

        # Determine output channels (specific to convnext_tiny, needs generalization for other models)
        # For convnext_tiny: dims are [96, 192, 384, 768] for stages 0, 1, 2, 3 respectively.
        # timm_model.feature_info.channels() is a reliable way if available
        try:
            self._out_channels = self.timm_model.feature_info.channels()
        except AttributeError:
            logger.warning(
                f"'feature_info' not found on {model_name}. Falling back to manual channel definition for convnext_tiny.")
            if 'convnext_tiny' in model_name:
                all_stage_channels = [96, 192, 384, 768]
                self._out_channels = [all_stage_channels[i]
                                      for i in out_indices]
            else:
                # This part would require a more robust way to get channel info for generic timm models
                # For now, we'll attempt a dummy forward pass if feature_info is missing
                try:
                    dummy_input = torch.randn(
                        1, 3, 224, 224)  # Standard image size
                    features = self.timm_model(dummy_input)
                    self._out_channels = [f.shape[1] for f in features]
                except Exception as e:
                    logger.error(
                        f"Could not determine output channels for {model_name} via dummy forward: {e}. Please check 'out_indices' and FPN 'in_channels' manually.")
                    self._out_channels = []  # Fallback, will likely cause issues downstream

        if not self._out_channels and out_indices == (3,) and 'convnext_tiny' in model_name:
            self._out_channels = [768]  # common case for bevformer_tiny
            logger.info(
                "Manually set out_channels to [768] for convnext_tiny with out_indices=(3,)")

    def forward(self, x):
        features = self.timm_model(x)
        # timm models with features_only=True and out_indices typically return a list of tensors
        # Ensure it's always a list for consistency with MMDetection's FPN
        if not isinstance(features, (list, tuple)):
            features = [features]
        return tuple(features)  # FPN expects a tuple

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # ConvNeXt uses LayerNorm, which typically does not have different
                # behavior for train/eval if track_running_stats is False (default for LN).
                # If a timm model internally uses BatchNorm, this would be relevant.
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    # init_weights is usually handled by timm's pretrained=True or by _load_checkpoint
    # def init_weights(self):
    #     pass # Covered by timm's pretrained or custom checkpoint loading in __init__

    @property
    def out_channels(self):
        # Property to be compatible with some MMDetection neck/head components
        # that might expect backbone.out_channels directly
        return self._out_channels
