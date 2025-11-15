# 深度学习模型集成说明

## 概述

本项目已集成基于 PyTorch 的深度学习模型（ResNet/EfficientNet），用于提高食物识别准确度。

## 安装依赖

```bash
pip install torch torchvision
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 模型架构

- **骨干网络**：支持 ResNet (18/34/50) 和 EfficientNet (B0/B1)
- **分类头**：自定义全连接层，输出类别数等于标签数量
- **预训练权重**：默认使用 ImageNet 预训练权重

## 配置

在 `server/.env` 文件中添加以下配置：

```env
# 模型类型（classifier|detector|hybrid）
MODEL_TYPE=classifier

# 模型骨干网络（resnet18|resnet34|resnet50|efficientnet_b0|efficientnet_b1）
MODEL_BACKBONE=resnet18

# 模型权重路径（可选，如果提供会加载自定义权重）
MODEL_PATH=

# 运行设备（cpu|cuda）
DEVICE=cpu
```

## 使用方式

### 1. 使用预训练 ImageNet 权重（默认）

不提供 `MODEL_PATH` 时，系统会使用 ImageNet 预训练的权重。虽然这些权重不是专门针对食物分类训练的，但可以作为特征提取器使用，比基于规则的分类更准确。

### 2. 使用自定义训练权重

如果你有自己的训练好的模型权重：

1. 将权重文件（`.pth` 或 `.pt`）放在服务器可访问的位置
2. 在 `.env` 中设置 `MODEL_PATH` 指向权重文件路径
3. 确保权重文件的格式与 `FoodClassifier` 模型结构匹配

### 3. 训练自定义模型

如果需要训练针对食物分类的模型：

1. **准备数据集**：
   - 收集各类食物的图片
   - 按照标签组织图片（每个标签一个文件夹）
   - 建议每个类别至少 100-200 张图片

2. **训练脚本示例**（需要单独创建）：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from server.services.model_service import FoodClassifier

# 加载模型
model = FoodClassifier(num_classes=20, backbone='resnet18', pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'labels': labels_list
}, 'food_classifier.pth')
```

## 回退机制

如果 PyTorch 未安装或模型加载失败，系统会自动回退到基于规则的分类方法，确保服务可用性。

## 性能优化

- **CPU 模式**：适合小规模使用，推理速度较慢
- **GPU 模式**：如果有 CUDA 设备，设置 `DEVICE=cuda` 可大幅提升推理速度
- **模型选择**：
  - `resnet18`：速度快，准确度中等，适合 CPU
  - `resnet50`：准确度高，速度较慢，适合 GPU
  - `efficientnet_b0`：准确度和速度平衡，推荐

## 注意事项

1. 首次运行时会自动下载 ImageNet 预训练权重（约 50-200MB）
2. 如果网络较慢，可能需要等待下载完成
3. 模型加载需要一定内存（ResNet18 约 50MB，ResNet50 约 100MB）
4. 使用 GPU 需要安装 CUDA 版本的 PyTorch

## 故障排查

### 问题：模型服务初始化失败

**原因**：PyTorch 未安装或版本不兼容

**解决**：
```bash
pip install torch torchvision
```

### 问题：CUDA 不可用

**原因**：未安装 CUDA 版本的 PyTorch 或没有 GPU

**解决**：使用 CPU 模式，设置 `DEVICE=cpu`

### 问题：识别准确度仍然不高

**原因**：使用 ImageNet 预训练权重，不是专门针对食物分类训练的

**解决**：
1. 收集食物图片数据集
2. 使用迁移学习微调模型
3. 训练完成后保存权重并在配置中指定路径

## 下一步

- [ ] 收集食物分类数据集
- [ ] 实现训练脚本
- [ ] 微调模型并评估性能
- [ ] 部署训练好的模型权重

