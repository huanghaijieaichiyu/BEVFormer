import torch
import torch.nn as nn
import timm
from mmcv.runner import BaseModule, load_checkpoint
# from mmcv.cnn import build_norm_layer # Not strictly needed if timm handles norms
from mmdet.models.builder import BACKBONES
# from mmengine.logging import MMLogger # Use get_logger for older mmcv/mmdet
from mmcv.utils import get_logger


@BACKBONES.register_module()
class TimmBackbone(BaseModule):
    """Generic backbone based on timm library.

    Args:
        model_name (str): Name of the model in timm.
        pretrained (bool): Whether to load timm's pretrained weights.
            Defaults to True.
        out_indices (tuple[int]): Output from which stages.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        layer_scale_init_value (float): Init value for Layer Scale (for specific models like ConvNeXt).
            Defaults to 1e-6.
        norm_eval (bool): Whether to set norm layers to eval mode.
            Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 model_name,
                 pretrained=True,
                 out_indices=(3,),
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6,
                 norm_eval=True,
                 init_cfg=None):
        effective_init_cfg = init_cfg
        if pretrained:
            effective_init_cfg = None

        super().__init__(init_cfg=effective_init_cfg)

        logger = get_logger('mmdet')

        self.norm_eval = norm_eval

        if not isinstance(out_indices, (list, tuple)):
            out_indices = [out_indices]

        # Create timm model
        # Pass only relevant kwargs to timm.create_model
        timm_kwargs = {}
        # Some models like convnext use ls_init_value, others don't.
        # A more robust way is to inspect the model's args, but for now we pass it.
        if 'convnext' in model_name or 'seresnext' in model_name:  # Add other models that use it
            timm_kwargs['ls_init_value'] = layer_scale_init_value

        self.timm_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            drop_path_rate=drop_path_rate,
            **timm_kwargs,
        )

        if not pretrained and init_cfg and init_cfg.get('type') == 'Pretrained' and init_cfg.get('checkpoint'):
            checkpoint_path = init_cfg['checkpoint']
            logger.info(
                f"Loading custom pretrained weights for {model_name} from {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

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
        try:
            self._out_channels = self.timm_model.feature_info.channels()
        except AttributeError:
            logger.warning(
                f"'feature_info' not found on {model_name}. Attempting to determine output channels via dummy forward pass.")
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.timm_model(dummy_input)
                self._out_channels = [f.shape[1] for f in features]
            except Exception as e:
                logger.error(
                    f"Could not determine output channels for {model_name} via dummy forward: {e}. Please check 'out_indices' and FPN 'in_channels' manually.")
                self._out_channels = []

    def forward(self, x):
        features = self.timm_model(x)
        if not isinstance(features, (list, tuple)):
            features = [features]
        return tuple(features)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    @property
    def out_channels(self):
        return self._out_channels
