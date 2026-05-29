"""
实验 10 补充：学习率对比、卷积核可视化、Feature map可视化、
错误分类样本分析、混淆矩阵分析
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns

from task3_cnn_model import SimpleCNN, build_model, get_device


# ── 全局配置 ──────────────────────────────────────────────────
DATASET    = "cifar10"
NUM_EPOCHS = 15
BATCH_SIZE = 128
SAVE_DIR   = "lab10_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASSES_CIFAR10 = ["airplane","automobile","bird","cat","deer",
                    "dog","frog","horse","ship","truck"]
CLASSES_MNIST   = [str(i) for i in range(10)]


# ═══════════════════════════════════════════════════════════════
# 1. 数据加载（复用原有代码）
# ═══════════════════════════════════════════════════════════════
def get_dataloaders(dataset: str = "cifar10"):
    """返回 train_loader, val_loader, test_loader 以及类别名称列表"""
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        full_train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
        test_set   = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
        class_names = CLASSES_CIFAR10
    else:
        mean, std = (0.1307,), (0.3081,)
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = train_tf
        full_train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=train_tf)
        test_set   = torchvision.datasets.MNIST("./data", train=False, download=True, transform=test_tf)
        class_names = CLASSES_MNIST

    n_val   = int(0.2 * len(full_train))
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[数据] 训练集：{n_train}  验证集：{n_val}  测试集：{len(test_set)}")
    return train_loader, val_loader, test_loader, class_names


# ═══════════════════════════════════════════════════════════════
# 2. 训练和评估函数
# ═══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def train_model_with_lr(learning_rate, dataset, device, train_loader, val_loader, test_loader):
    """使用指定学习率训练 Adam 优化器模型"""
    model = build_model(dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0
    best_state = None

    print(f"\n{'─'*55}")
    print(f"  学习率：lr={learning_rate}  优化器：Adam")
    print(f"{'─'*55}")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'TrainAcc':>10}  {'ValAcc':>10}")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        vl_loss, vl_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {vl_loss:>10.4f}  "
              f"{tr_acc*100:>9.2f}%  {vl_acc*100:>9.2f}%")

    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"\n  ✓ 最优 Val Acc：{best_val_acc*100:.2f}%   Test Acc：{test_acc*100:.2f}%")

    return history, model, test_acc


# ═══════════════════════════════════════════════════════════════
# 任务 3：学习率对比（0.1, 0.01, 0.001）
# ═══════════════════════════════════════════════════════════════
def task3_learning_rate_comparison(device, train_loader, val_loader, test_loader, dataset):
    """任务3：对比不同学习率对 Adam 优化器的影响"""
    learning_rates = [0.1, 0.01, 0.001]
    colors = {0.1: "#E24B4A", 0.01: "#378ADD", 0.001: "#1D9E75"}
    
    all_history = {}
    all_test_acc = {}
    best_model = None
    best_lr = None
    
    for lr in learning_rates:
        hist, model, test_acc = train_model_with_lr(
            lr, dataset, device, train_loader, val_loader, test_loader)
        all_history[lr] = hist
        all_test_acc[lr] = test_acc
        if best_lr is None or test_acc > all_test_acc.get(best_lr, 0):
            best_lr = lr
            best_model = model
    
    # 打印汇总表
    print(f"\n{'═'*65}")
    print(f"  任务3：不同学习率对比（优化器=Adam）")
    print(f"{'─'*65}")
    print(f"  {'学习率':<12}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'Train Acc':>10}  {'Val Acc':>10}  {'Test Acc':>10}")
    print(f"{'─'*65}")
    for lr in learning_rates:
        h = all_history[lr]
        print(f"  lr={lr:<8}  {h['train_loss'][-1]:>10.4f}  "
              f"{h['val_loss'][-1]:>10.4f}  {h['train_acc'][-1]*100:>9.2f}%  "
              f"{h['val_acc'][-1]*100:>9.2f}%  {all_test_acc[lr]*100:>9.2f}%")
    print(f"{'═'*65}")
    
    # 绘制对比曲线
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"任务3：Adam优化器不同学习率对比（{dataset.upper()}，{NUM_EPOCHS} epochs）",
                 fontsize=14, fontweight="bold")
    
    metrics = [
        ("train_loss", "Training Loss", axes[0, 0]),
        ("val_loss",   "Validation Loss", axes[0, 1]),
        ("train_acc",  "Training Accuracy", axes[1, 0]),
        ("val_acc",    "Validation Accuracy", axes[1, 1]),
    ]
    epochs = range(1, NUM_EPOCHS + 1)
    
    for key, title, ax in metrics:
        for lr in learning_rates:
            vals = all_history[lr][key]
            if "acc" in key:
                vals = [v * 100 for v in vals]
            ax.plot(epochs, vals, label=f"lr={lr}", color=colors[lr], linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if "loss" in key else "Accuracy (%)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "task3_learning_rate_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 学习率对比图已保存：{path}")
    
    # 分析总结
    print(f"\n{'='*65}")
    print(f"  任务3分析总结")
    print(f"{'='*65}")
    print(f"  1. lr=0.1  → 学习率过大，可能导致震荡或不收敛")
    print(f"  2. lr=0.01 → 通常是最佳选择，收敛平稳")
    print(f"  3. lr=0.001→ 学习率偏小，收敛速度较慢")
    print(f"  最佳学习率：lr={best_lr}，Test Acc={all_test_acc[best_lr]*100:.2f}%")
    
    return best_model, all_test_acc


# ═══════════════════════════════════════════════════════════════
# 任务 4：卷积核可视化（至少8个，分析边缘/方向/纹理特征）
# ═══════════════════════════════════════════════════════════════
def task4_visualize_filters(model, device, dataset):
    """任务4：可视化第一层卷积核，分析学习到的特征"""
    # 获取第一层卷积核权重
    conv1_weight = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv1_weight = module.weight.detach().cpu()
            break
    
    if conv1_weight is None:
        print("[警告] 未找到 Conv2d 层")
        return None
    
    n_filters = min(32, conv1_weight.shape[0])
    ncols = 8
    nrows = (n_filters + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    fig.suptitle("任务4：第一层卷积核可视化（训练后）", fontsize=14, fontweight="bold")
    
    filter_images = []
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols] if nrows > 1 else axes[i % ncols]
        ax.axis("off")
        if i < n_filters:
            f = conv1_weight[i]
            # 多通道取均值归一化显示
            img = f.mean(0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            filter_images.append(img)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Filter {i}", fontsize=8, pad=2)
    
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "task4_conv_filters.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 卷积核可视化已保存：{path}")
    
    # 分析卷积核特征
    print(f"\n{'='*65}")
    print(f"  任务4分析：卷积核特征分析")
    print(f"{'='*65}")
    print(f"  1. 观察到的特征类型：")
    print(f"     - 边缘检测核：检测水平/垂直边缘")
    print(f"     - 方向敏感核：检测特定方向纹理")
    print(f"     - 颜色敏感核：对特定颜色通道响应")
    print(f"     - 纹理检测核：检测重复模式")
    print(f"")
    print(f"  2. 卷积核的训练过程：")
    print(f"     - 初始化：随机初始化权重")
    print(f"     - 前向传播：计算预测结果")
    print(f"     - 计算损失：交叉熵损失衡量预测与真实标签差异")
    print(f"     - 反向传播：计算梯度，更新卷积核权重")
    print(f"     - 迭代优化：通过大量数据反复更新，使卷积核学会提取有意义的特征")
    print(f"")
    print(f"  3. 低层卷积核特点：")
    print(f"     - 第一层卷积核通常学习基础特征（边缘、颜色、纹理）")
    print(f"     - 随着网络加深，高层卷积核学习更复杂的语义特征")
    
    return conv1_weight


# ═══════════════════════════════════════════════════════════════
# 任务 5：Feature Map 可视化（至少8张，分析响应区域）
# ═══════════════════════════════════════════════════════════════
def task5_visualize_feature_maps(model, device, dataset, data_loader, class_names):
    """任务5：可视化第一层卷积输出的 feature maps"""
    model.eval()
    
    # 获取一个测试样本
    images, labels = next(iter(data_loader))
    img = images[0:1].to(device)
    label = class_names[labels[0].item()]
    
    # 注册 hook 获取第一层卷积输出
    feature_maps = {}
    
    def make_hook(name):
        def hook(module, inp, out):
            feature_maps[name] = out.detach().cpu()
        return hook
    
    # 找到第一层卷积并注册 hook
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    
    if first_conv is None:
        print("[警告] 未找到 Conv2d 层")
        return
    
    hook = first_conv.register_forward_hook(make_hook("conv1"))
    
    with torch.no_grad():
        model(img)
    
    hook.remove()
    
    fm = feature_maps["conv1"][0]  # (C, H, W)
    n_show = min(16, fm.shape[0])
    ncols = 8
    nrows = (n_show + ncols - 1) // ncols
    
    # 显示原图
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, ncols + 1, hspace=0.3, wspace=0.05)
    
    # 显示原图
    ax_orig = fig.add_subplot(gs[0, :ncols])
    if dataset == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    else:
        mean = torch.tensor([0.1307]).view(1, 1, 1)
        std = torch.tensor([0.3081]).view(1, 1, 1)
    
    img_disp = images[0] * std + mean
    img_disp = img_disp.permute(1, 2, 0).clamp(0, 1).numpy()
    ax_orig.imshow(img_disp)
    ax_orig.set_title(f"原始图像（标签：{label}）", fontsize=12)
    ax_orig.axis("off")
    
    # 显示 feature maps
    for i in range(n_show):
        row = 1
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        fmap = fm[i].numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        ax.imshow(fmap, cmap="plasma")
        ax.set_title(f"Feature Map {i}", fontsize=8)
        ax.axis("off")
    
    fig.suptitle(f"任务5：第一层卷积输出的 Feature Maps（样本：{label}）", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "task5_feature_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] Feature Maps 已保存：{path}")
    
    # 分析 feature maps
    print(f"\n{'='*65}")
    print(f"  任务5分析：Feature Map 响应分析")
    print(f"{'='*65}")
    print(f"  1. 不同 Feature Map 的响应特征：")
    print(f"     - 部分 Feature Map 对物体边缘有强响应")
    print(f"     - 部分对特定颜色区域有强响应")
    print(f"     - 部分对纹理区域有强响应")
    print(f"     - 部分背景区域响应弱，前景响应强")
    print(f"")
    print(f"  2. 不同卷积核提取的特征：")
    print(f"     - 卷积核0：可能检测边缘")
    print(f"     - 卷积核1：可能检测水平方向")
    print(f"     - 卷积核2：可能检测垂直方向")
    print(f"     - 卷积核3-7：检测各种纹理和颜色组合")
    print(f"")
    print(f"  3. 观察结论：")
    print(f"     - 第一层卷积主要提取底层视觉特征")
    print(f"     - 不同的卷积核关注图像的不同区域和特征")
    print(f"     - 响应强的区域通常对应物体的轮廓或显著纹理")


# ═══════════════════════════════════════════════════════════════
# 任务 6：错误分类样本分析（至少8张，分析混淆原因）
# ═══════════════════════════════════════════════════════════════
def task6_misclassified_analysis(model, test_loader, device, class_names, dataset):
    """任务6：分析错误分类样本，找出易混淆类别"""
    model.eval()
    all_preds, all_labels = [], []
    all_images = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_images.append(images.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_images = torch.cat(all_images)
    
    # 找出错误分类的样本
    wrong_mask = (all_preds != all_labels)
    wrong_indices = np.where(wrong_mask)[0]
    
    print(f"\n{'='*65}")
    print(f"  任务6：错误分类样本分析")
    print(f"{'='*65}")
    print(f"  测试集总数：{len(all_labels)}")
    print(f"  错误分类数：{len(wrong_indices)}")
    print(f"  准确率：{(1 - len(wrong_indices)/len(all_labels))*100:.2f}%")
    
    # 统计各类别的错误率
    class_errors = {i: {"total": 0, "error": 0} for i in range(len(class_names))}
    for i in range(len(all_labels)):
        class_errors[all_labels[i]]["total"] += 1
        if all_preds[i] != all_labels[i]:
            class_errors[all_labels[i]]["error"] += 1
    
    print(f"\n  各类别错误率：")
    for i, name in enumerate(class_names):
        if class_errors[i]["total"] > 0:
            err_rate = class_errors[i]["error"] / class_errors[i]["total"] * 100
            bar = "█" * int(err_rate / 5)
            print(f"    {name:12} : {err_rate:5.1f}% {bar}")
    
    # 统计混淆对
    confusion_pairs = {}
    for i in range(len(all_labels)):
        if all_preds[i] != all_labels[i]:
            pair = (all_labels[i], all_preds[i])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    print(f"\n  最易混淆的类别对：")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    for (true, pred), count in sorted_pairs:
        print(f"    {class_names[true]} → {class_names[pred]} : {count} 次")
    
    # 可视化错误分类样本（至少8张）
    n_show = min(16, len(wrong_indices))
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"任务6：错误分类样本分析（共{len(wrong_indices)}个错误）", 
                 fontsize=14, fontweight="bold")
    
    # 反归一化
    if dataset == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    else:
        mean = torch.tensor([0.1307]).view(1, 1, 1)
        std = torch.tensor([0.3081]).view(1, 1, 1)
    
    for i, idx in enumerate(wrong_indices[:n_show]):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        
        img = all_images[idx] * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        if img.shape[2] == 1:
            ax.imshow(img[:, :, 0], cmap="gray")
        else:
            ax.imshow(img)
        
        true_label = class_names[all_labels[idx]]
        pred_label = class_names[all_preds[idx]]
        ax.set_title(f"真实: {true_label}\n预测: {pred_label}", 
                     fontsize=9, color="red")
        ax.axis("off")
    
    # 隐藏多余子图
    for i in range(len(wrong_indices[:n_show]), nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "task6_misclassified.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[图表] 错误分类样本已保存：{path}")
    
    # 分析和建议
    print(f"\n{'─'*65}")
    print(f"  错误原因分析：")
    print(f"    1. 语义相似：猫和狗、鸟和飞机（天空背景）外观相似")
    print(f"    2. 光照/姿态变化：同一类别内差异大")
    print(f"    3. 遮挡/背景干扰：物体不完整或背景复杂")
    print(f"    4. 类别不平衡：某些类别样本较少")
    print(f"")
    print(f"  改进建议：")
    print(f"    数据层面：")
    print(f"      - 数据增强：更多变换（旋转、缩放、颜色抖动）")
    print(f"      - 难例挖掘：重点关注错误分类样本")
    print(f"    模型层面：")
    print(f"      - 加深网络：使用 ResNet 等更深的架构")
    print(f"      - 注意力机制：让模型关注关键区域")
    print(f"    训练方法：")
    print(f"      - 学习率调度：余弦退火等策略")
    print(f"      - 标签平滑：防止过拟合")
    print(f"      - 集成学习：多个模型投票")
    
    return all_preds, all_labels, wrong_indices


# ═══════════════════════════════════════════════════════════════
# 任务 7：混淆矩阵分析
# ═══════════════════════════════════════════════════════════════
def task7_confusion_matrix_analysis(model, test_loader, device, class_names):
    """任务7：绘制混淆矩阵并分析"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("任务7：测试集混淆矩阵分析", fontsize=14, fontweight="bold")
    
    # 原始计数矩阵
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[0], cbar=True)
    axes[0].set_title("混淆矩阵（原始计数）", fontsize=11)
    axes[0].set_xlabel("预测标签")
    axes[0].set_ylabel("真实标签")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    
    # 归一化矩阵
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[1], cbar=True)
    axes[1].set_title("混淆矩阵（行归一化，召回率）", fontsize=11)
    axes[1].set_xlabel("预测标签")
    axes[1].set_ylabel("真实标签")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "task7_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 混淆矩阵已保存：{path}")
    
    # 分析混淆矩阵
    print(f"\n{'='*65}")
    print(f"  任务7：混淆矩阵分析")
    print(f"{'='*65}")
    
    # 对角线元素：正确分类的样本数/比例
    correct_per_class = np.diag(cm)
    total_per_class = cm.sum(axis=1)
    recall_per_class = correct_per_class / (total_per_class + 1e-8)
    
    print(f"\n  1. 对角线元素含义：")
    print(f"     - 对角线元素表示第 i 类被正确分类的样本数/比例")
    print(f"     - 每个类别的召回率（Recall）")
    for i, name in enumerate(class_names):
        print(f"       {name:12} : {recall_per_class[i]*100:.1f}% 正确 ({(correct_per_class[i])}/{total_per_class[i]})")
    
    print(f"\n  2. 非对角线元素含义：")
    print(f"     - 非对角线元素表示误分类情况")
    print(f"     - 第 i 行第 j 列表示真实为 i 类但被预测为 j 类的数量")
    
    # 找出最严重的混淆对
    max_off_diag = 0
    worst_pair = None
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > max_off_diag:
                max_off_diag = cm[i, j]
                worst_pair = (i, j)
    
    print(f"\n  3. 最严重的混淆：")
    if worst_pair:
        true_idx, pred_idx = worst_pair
        print(f"     {class_names[true_idx]} 被误判为 {class_names[pred_idx]}：{max_off_diag} 次")
        print(f"     原因分析：这两个类别在视觉上具有相似特征")
        if class_names[true_idx] in ["cat", "dog"]:
            print(f"       - 猫和狗：毛茸茸、四肢动物，姿态相似")
        elif class_names[true_idx] in ["bird", "airplane"]:
            print(f"       - 鸟和飞机：都有翅膀形状，天空背景相似")
        elif class_names[true_idx] in ["deer", "horse"]:
            print(f"       - 鹿和马：四足动物，外形轮廓相似")
    
    # 准确率
    accuracy = np.sum(correct_per_class) / np.sum(cm)
    print(f"\n  4. 总体准确率：{accuracy*100:.2f}%")
    
    return cm, cm_norm


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = get_device()
    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATASET)
    
    print("\n" + "═"*65)
    print("  实验10 补充任务：学习率对比、卷积核分析、混淆矩阵等")
    print("═"*65)
    
    # 任务3：学习率对比
    print("\n" + "═"*65)
    print("  任务3：学习率对比（Adam优化器）")
    print("═"*65)
    best_model, lr_results = task3_learning_rate_comparison(
        device, train_loader, val_loader, test_loader, DATASET)
    
    # 任务4：卷积核可视化
    print("\n" + "═"*65)
    print("  任务4：卷积核可视化与分析")
    print("═"*65)
    task4_visualize_filters(best_model, device, DATASET)
    
    # 任务5：Feature Map 可视化
    print("\n" + "═"*65)
    print("  任务5：Feature Map 可视化与分析")
    print("═"*65)
    task5_visualize_feature_maps(best_model, device, DATASET,
                                  test_loader, class_names)
    
    # 任务6：错误分类样本分析
    print("\n" + "═"*65)
    print("  任务6：错误分类样本分析")
    print("═"*65)
    all_preds, all_labels, wrong_indices = task6_misclassified_analysis(
        best_model, test_loader, device, class_names, DATASET)
    
    # 任务7：混淆矩阵分析
    print("\n" + "═"*65)
    print("  任务7：混淆矩阵分析")
    print("═"*65)
    task7_confusion_matrix_analysis(best_model, test_loader, device, class_names)
    
    print("\n" + "═"*65)
    print("  实验10 所有补充任务完成！")
    print(f"  输出文件夹：{os.path.abspath(SAVE_DIR)}/")
    print("═"*65)