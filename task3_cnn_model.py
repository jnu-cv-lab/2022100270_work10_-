

"""
实验 8：PyTorch 入门与图像分类
任务 3：CNN 模型设计

支持数据集：MNIST (1×28×28, 10类) / CIFAR-10 (3×32×32, 10类)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 3.1  模型定义
# ──────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    简单 CNN 图像分类模型

    网络结构（以 CIFAR-10 为例）：
        输入  (B, C, H, W)
          │
          ├─ 卷积块 1：Conv2d → BN → ReLU → MaxPool
          ├─ 卷积块 2：Conv2d → BN → ReLU → MaxPool
          ├─ 卷积块 3：Conv2d → BN → ReLU → MaxPool
          │
          ├─ AdaptiveAvgPool  →  将特征图压缩为 4×4
          ├─ Flatten
          │
          ├─ 全连接层 1：Linear → ReLU → Dropout
          ├─ 全连接层 2：Linear → ReLU → Dropout
          └─ 输出层   ：Linear  (logits, 不含 Softmax)

    参数
    ------
    in_channels : int
        输入通道数。MNIST=1，CIFAR-10=3
    num_classes : int
        输出类别数。MNIST/CIFAR-10 均为 10
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # ── 卷积块 1 ──────────────────────────────────
        # 卷积层：提取低级特征（边缘、纹理）
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32,            # 输入通道 → 32 个滤波器
                      kernel_size=3, padding=1),  # same padding 保持尺寸
            nn.BatchNorm2d(32),                   # 批归一化，加速收敛
            nn.ReLU(inplace=True),                # 激活函数：ReLU
            nn.MaxPool2d(kernel_size=2, stride=2) # 池化层：尺寸减半
        )

        # ── 卷积块 2 ──────────────────────────────────
        # 64 个滤波器，提取更高层次特征
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── 卷积块 3 ──────────────────────────────────
        # 128 个滤波器，提取更抽象语义特征
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── 自适应平均池化 ─────────────────────────────
        # 无论输入图像尺寸如何，统一输出 4×4 特征图
        # 这使模型同时兼容 MNIST(28×28) 和 CIFAR-10(32×32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── 全连接分类头 ───────────────────────────────
        self.classifier = nn.Sequential(
            # 全连接层 1
            nn.Flatten(),                         # 展平：128×4×4 = 2048
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                    # Dropout 防止过拟合

            # 全连接层 2
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # 输出层：输出 logits（CrossEntropyLoss 内含 Softmax）
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv_block1(x)     # 特征提取 Block 1
        x = self.conv_block2(x)     # 特征提取 Block 2
        x = self.conv_block3(x)     # 特征提取 Block 3
        x = self.adaptive_pool(x)   # 自适应池化
        x = self.classifier(x)      # 全连接分类
        return x                    # 返回 logits


# ──────────────────────────────────────────────
# 3.2  模型实例化 & 验证
# ──────────────────────────────────────────────
def build_model(dataset: str = "cifar10") -> SimpleCNN:
    """
    根据数据集名称创建模型

    参数
    ------
    dataset : str
        "mnist" 或 "cifar10"

    返回
    ------
    model : SimpleCNN
    """
    config = {
        "mnist":   {"in_channels": 1, "num_classes": 10},
        "cifar10": {"in_channels": 3, "num_classes": 10},
    }
    assert dataset in config, f"不支持的数据集：{dataset}，请选择 'mnist' 或 'cifar10'"

    model = SimpleCNN(**config[dataset])
    return model


def print_model_summary(model: SimpleCNN, dataset: str = "cifar10"):
    """打印模型结构与参数统计"""
    input_shape = {
        "mnist":   (1, 1, 28, 28),
        "cifar10": (1, 3, 32, 32),
    }

    print("=" * 60)
    print(f"  SimpleCNN 模型结构（数据集：{dataset.upper()}）")
    print("=" * 60)
    print(model)
    print()

    # 逐层特征图尺寸追踪
    device = next(model.parameters()).device
    dummy = torch.zeros(*input_shape[dataset]).to(device)
    model.eval()
    with torch.no_grad():
        x = dummy
        print(f"{'层名':<22} {'输入尺寸':<22} {'输出尺寸'}")
        print("-" * 65)
        print(f"{'Input':<22} {'-':<22} {str(list(x.shape))}")
        x = model.conv_block1(x)
        print(f"{'conv_block1':<22} {str(list(dummy.shape)):<22} {str(list(x.shape))}")
        prev = x
        x = model.conv_block2(x)
        print(f"{'conv_block2':<22} {str(list(prev.shape)):<22} {str(list(x.shape))}")
        prev = x
        x = model.conv_block3(x)
        print(f"{'conv_block3':<22} {str(list(prev.shape)):<22} {str(list(x.shape))}")
        prev = x
        x = model.adaptive_pool(x)
        print(f"{'adaptive_pool':<22} {str(list(prev.shape)):<22} {str(list(x.shape))}")
        prev = x
        x = model.classifier(x)
        print(f"{'classifier (FC+Out)':<22} {str(list(prev.shape)):<22} {str(list(x.shape))}")

    # 参数量统计
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 65)
    print(f"  总参数量：    {total:,}")
    print(f"  可训练参数：  {trainable:,}")
    print("=" * 60)


# ──────────────────────────────────────────────
# 3.3  设备选择工具
# ──────────────────────────────────────────────
def get_device() -> torch.device:
    """自动选择 GPU / CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用：{device}")
    return device


# ──────────────────────────────────────────────
# 3.4  主程序入口（任务 3 验证）
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ── 选择数据集（与任务 2 保持一致）─────────────────
    DATASET = "cifar10"   # 可改为 "mnist"

    device = get_device()

    # 构建模型并移至设备
    model = build_model(DATASET).to(device)

    # 打印结构摘要
    print_model_summary(model, DATASET)

    # 随机输入前向传播验证
    input_shape = (4, 3, 32, 32) if DATASET == "cifar10" else (4, 1, 28, 28)
    dummy_input = torch.randn(*input_shape).to(device)
    output = model(dummy_input)
    print(f"\n[验证] 输入形状：{list(dummy_input.shape)}")
    print(f"[验证] 输出形状：{list(output.shape)}  ← (batch_size, num_classes)")
    print(f"[验证] 模型构建成功！✓")

    # ── 与任务 2 数据加载器对接示例 ─────────────────────
    # （假设 train_loader / val_loader / test_loader 已在任务 2 中定义）
    #
    # loss_fn  = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #
    # for images, labels in train_loader:
    #     images, labels = images.to(device), labels.to(device)
    #     outputs = model(images)          # 前向传播
    #     loss = loss_fn(outputs, labels)  # 计算损失
    #     optimizer.zero_grad()
    #     loss.backward()                  # 反向传播
    #     optimizer.step()

