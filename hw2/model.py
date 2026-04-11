import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    基础版 CNN：
    适合作为作业中的 baseline 模型
    输入: [B, 3, 32, 32]
    输出: [B, 10]
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)    # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EnhancedCNN(nn.Module):
    """
    增强版 CNN：
    用于高分作业版本
    加入了：
    1. 更深的卷积层
    2. BatchNorm
    3. Dropout
    4. AdaptiveAvgPool2d 降低全连接层参数量
    """
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 -> 16
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),   # 16 -> 8
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),   # 8 -> 4
            nn.Dropout(0.3),

            # 统一输出为 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name="enhanced", num_classes=10):
    """
    根据名称返回模型
    """
    model_name = model_name.lower()

    if model_name == "simple":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "enhanced":
        return EnhancedCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


if __name__ == "__main__":
    # 简单测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(8, 3, 32, 32).to(device)

    model1 = SimpleCNN().to(device)
    y1 = model1(x)
    print("SimpleCNN output shape:", y1.shape)

    model2 = EnhancedCNN().to(device)
    y2 = model2(x)
    print("EnhancedCNN output shape:", y2.shape)