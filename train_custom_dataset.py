# -*- coding: utf-8 -*-
"""
自定义数据集训练脚本
使用指定路径的数据集训练 EfficientNet-B0 模型
"""
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
import os
import math
import random
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import urllib.error

# ----------------------------
# 超参数
# ----------------------------
DATASET_DIR = r"C:\Users\33106\Desktop\每个类别提供5张的部分数据集\每个类别提供5张的部分数据集"

BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 2e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 8        # 连续多少轮 val 准确率不提升则早停
IMG_SIZE     = 224
NUM_WORKERS  = 0        # Windows 建议 0；Linux 可设 4/8
SEED         = 42

# 若你有本地的 EfficientNet-B0 权重（state_dict），可填入路径；否则保持 None
LOCAL_WEIGHTS_PATH = None  # 例如 r"C:\Users\33106\Desktop\shipin\efficientnet_b0.pth"

# ----------------------------
# 随机种子
# ----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# 模型构建（自动回退）
# ----------------------------
def build_model(num_classes, try_pretrained=True, local_weights_path=None):
    """
    - local_weights_path 优先：完全离线，用你自己的 .pth
    - try_pretrained=True：尝试从 torchvision 加载 ImageNet 预训练（联网），失败则自动回退到随机初始化
    """
    if local_weights_path is not None:
        print(f"🟡 使用本地权重：{local_weights_path}")
        model = efficientnet_b0(weights=None)
        state = torch.load(local_weights_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("✅ 已加载本地权重 .pth")
    elif try_pretrained:
        try:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
            print("✅ 已加载 ImageNet 预训练权重")
        except (urllib.error.URLError, RuntimeError, OSError) as e:
            print(f"⚠️ 预训练权重下载失败，改为随机初始化：{e}")
            model = efficientnet_b0(weights=None)
    else:
        model = efficientnet_b0(weights=None)
    
    # 替换分类头
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# ----------------------------
# 数据增强 & 归一化
# ----------------------------
def build_transforms(img_size: int = 224):
    """
    通用且兼容各版本 torchvision 的预处理。
    评估/测试直接用官方权重的 transforms；
    训练在同样的 mean/std 上加轻量数据增强。
    """
    base = EfficientNet_B0_Weights.IMAGENET1K_V1
    mean = base.meta.get("mean", [0.485, 0.456, 0.406])
    std  = base.meta.get("std",  [0.229, 0.224, 0.225])

    # 评估/测试：官方预处理管线（已含 Resize/CenterCrop/ToTensor/Normalize）
    eval_tfms = base.transforms()

    # 训练：在相同几何尺度上增加轻量增强，再手动 Normalize
    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfms, eval_tfms

# ----------------------------
# 数据集准备
# 支持两种目录结构：
# 1) 根目录下已有 train/val/test 子文件夹
# 2) 根目录为类别文件夹集合（则自动 8:1:1 划分）
# ----------------------------
def prepare_dataloaders(root_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    root = Path(root_dir)
    assert root.exists(), f"数据集目录不存在：{root}"

    train_tfms, eval_tfms = build_transforms()

    subdirs = [p.name.lower() for p in root.iterdir() if p.is_dir()]

    has_split = all(d in subdirs for d in ["train", "val", "test"])

    if has_split:
        train_ds = ImageFolder(root / "train", transform=train_tfms)
        val_ds   = ImageFolder(root / "val",   transform=eval_tfms)
        test_ds  = ImageFolder(root / "test",  transform=eval_tfms)
        classes  = train_ds.classes
        print("✅ 使用现有的 train/val/test 目录")
    else:
        # 自动划分 8:1:1
        full_ds = ImageFolder(root, transform=train_tfms)
        classes = full_ds.classes

        n = len(full_ds)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        n_test  = n - n_train - n_val

        # 为了 val/test 不用训练增强，做两个"视图包装"
        idxs = list(range(n))
        random.shuffle(idxs)
        train_idx = idxs[:n_train]
        val_idx   = idxs[n_train:n_train+n_val]
        test_idx  = idxs[n_train+n_val:]

        train_ds = Subset(ImageFolder(root, transform=train_tfms), train_idx)

        # 包一层以改 val/test 的 transform
        val_base  = ImageFolder(root, transform=eval_tfms)
        test_base = ImageFolder(root, transform=eval_tfms)
        val_ds  = Subset(val_base,  val_idx)
        test_ds = Subset(test_base, test_idx)

        # 给 Subset 补 classes 属性（便于后续使用）
        train_ds.classes = classes
        val_ds.classes = classes
        test_ds.classes = classes

        print(f"✅ 自动划分数据集：train {n_train} / val {n_val} / test {n_test}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes

# ----------------------------
# 训练与验证
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss_sum += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    avg_loss = loss_sum/total
    acc = correct/total
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return avg_loss, acc, all_preds, all_labels

# ----------------------------
# 主流程
# ----------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"🚀 Device: {device}")

    # 检查数据集路径
    if not os.path.exists(DATASET_DIR):
        print(f"❌ 数据集目录不存在：{DATASET_DIR}")
        print("请检查路径是否正确")
        return

    train_loader, val_loader, test_loader, classes = prepare_dataloaders(DATASET_DIR, BATCH_SIZE, NUM_WORKERS)

    num_classes = len(classes)
    print(f"🔤 类别数：{num_classes}")
    print(f"📋 类别列表：{classes}")

    model = build_model(
        num_classes,
        try_pretrained=True,
        local_weights_path=LOCAL_WEIGHTS_PATH
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_acc = 0.0
    best_path = "best_model_custom.pth"
    no_improve = 0

    print("==== 开始训练 ====")
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        dt = time.time() - t0
        print(f"[{epoch:03d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} | {dt:.1f}s")

        # 保存最佳
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "num_classes": num_classes,
            }, best_path)
            print(f"  ✅ New best! 保存模型到 {best_path}  (val_acc={val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"⏹️ 早停：连续 {EARLY_STOP} 轮未提升")
                break

    # 加载最佳并在 test 上评估
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"📦 已加载最佳模型（val_acc={ckpt.get('val_acc',0):.4f}）进行测试评估")
    else:
        print("⚠️ 未找到最佳模型文件，使用当前权重进行测试评估")

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print("\n==== 测试集表现 ====")
    print(f"Test  Loss: {test_loss:.4f}")
    print(f"Test  Acc : {test_acc*100:.2f}%")  # <- 终端输出识别准确率

    try:
        print("\n分类报告：")
        print(classification_report(test_labels, test_preds, target_names=classes, digits=4))
    except Exception as e:
        print(f"分类报告生成失败：{e}")

    try:
        cm = confusion_matrix(test_labels, test_preds)
        print("\n混淆矩阵（行=真实/列=预测）：")
        print(cm)
    except Exception as e:
        print(f"混淆矩阵生成失败：{e}")

    # 保存类别映射到 label_map.json（供系统使用）
    import json
    label_map_path = "server/assets/label_map.json"
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存类别映射到 {label_map_path}")

    print("\n✅ 训练已完成。")
    print(f"📁 模型文件：{best_path}")
    print(f"📋 类别映射：{label_map_path}")

if __name__ == "__main__":
    # 可选：命令行参数覆盖（不想用可忽略）
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATASET_DIR, help="数据集根目录")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--local_weights", type=str, default=LOCAL_WEIGHTS_PATH)
    args = parser.parse_args()

    # 覆盖全局
    DATASET_DIR = args.data
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    LR = args.lr
    NUM_WORKERS = args.workers
    IMG_SIZE = args.imgsz
    LOCAL_WEIGHTS_PATH = args.local_weights if args.local_weights not in [None, "None", ""] else None

    main()

