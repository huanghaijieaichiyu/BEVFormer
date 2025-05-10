# BEVFormer知识蒸馏

本目录包含使用知识蒸馏从预训练的BEVFormer模型中提取知识的配置文件。

## 特征图蒸馏方案

当前实现了基于BEV特征图的蒸馏方案，使用教师模型（如`bevformer_tiny`）的BEV特征来指导学生模型（如`bevformer_pico`）的训练。

### 主要特点

- **BEV特征蒸馏**：捕获教师模型的BEV表示，并让学生模型学习相似的特征
- **特征适配器**：当教师和学生模型的特征维度不同时，使用1x1卷积或线性层进行适配
- **特征空间变换**：当教师和学生模型的BEV空间分辨率不同时，使用双线性插值进行调整
- **灵活的损失函数**：支持MSE和L1损失函数

## 使用方法

### 准备工作

1. 确保您有一个预训练好的教师模型（如`bevformer_tiny`）的权重文件
2. 修改配置文件中的`teacher_checkpoint_path`指向您的权重文件路径

### 训练命令

单GPU训练:
```bash
python tools/train.py projects/configs/bevformer_distill/bevformer_pico_from_tiny_distill.py --work-dir work_dirs/bevformer_pico_distill
```

多GPU训练:
```bash
bash tools/train_distill.sh projects/configs/bevformer_distill/bevformer_pico_from_tiny_distill.py 8
```

## 配置文件说明

### bevformer_pico_from_tiny_distill.py

该配置文件定义了:

1. 教师模型：使用`bevformer_tiny`作为教师
2. 学生模型：定义了一个更小的`bevformer_pico`模型
   - 更小的特征维度（128 vs 256）
   - 更小的BEV分辨率（32x32 vs 50x50）
   - 更少的Transformer层（编码器2层 vs 3层，解码器3层 vs 6层）
   - 更小的骨干网络（convnext_femto vs convnext_nano）
3. 蒸馏设置：
   - 损失类型：MSE
   - 特征适配：1x1卷积
   - 损失权重：1.0

## 自定义蒸馏

您可以通过修改以下参数来自定义蒸馏过程:

1. `distillation_cfg`中的参数:
   - `loss_weight`: 控制蒸馏损失的权重
   - `loss_type`: 选择`mse`或`l1`损失
   - `adapter`: 特征适配器类型，当教师和学生模型的特征维度不同时使用

2. 学生模型架构参数:
   - `_dim_`: 特征维度
   - `bev_h_`, `bev_w_`: BEV特征图大小
   - 编码器和解码器的层数
   - 骨干网络类型和参数

## 注意事项

- 确保教师模型已经在目标数据集上充分训练
- 学生模型应该比教师模型更小，以实现模型压缩
- 可能需要调整学习率和训练时间，因为蒸馏通常需要更长的训练时间 