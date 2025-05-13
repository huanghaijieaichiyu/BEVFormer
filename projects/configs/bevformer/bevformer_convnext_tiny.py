# bevformer_convnext_tiny.py

_base_ = ['bevformer_tiny.py']

# 使用 ConvNeXt-Pico 作为图像骨干
model = dict(
    # 清除基类中预训练字段
    pretrained=None,
    img_backbone=dict(
        _delete_=True,
        type='ConvNeXtTimm',  # use ConvNeXt wrapper based on timm
        model_name='convnext_pico',  # 输出通道数为 [96,192,384,768]
        pretrained=True,
        out_indices=(3,),  # 只使用最后阶段，通道数为768
        drop_path_rate=0.0,
        norm_eval=True,
    ),
    img_neck=dict(
        # ConvNeXt Pico 最后阶段输出通道数为 512
        in_channels=[512],
    ),
) 