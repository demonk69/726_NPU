# model.py
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class VGGLikeCNN(nn.Module):
    """
    输入:  [B, 3, 32, 32]
    输出:  [B, 10]
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # 32x32 -> 32x32 -> 16x16
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 16x16 -> 16x16 -> 8x8
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 8x8 -> 8x8 -> 4x4
        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 4x4 -> 4x4
        self.stage4 = nn.Sequential(
            ConvBNReLU(256, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stage1(x)   # [B, 64, 16, 16]
        x = self.stage2(x)   # [B, 128, 8, 8]
        x = self.stage3(x)   # [B, 256, 4, 4]
        x = self.stage4(x)   # [B, 512, 4, 4]
        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.classifier(x)   # [B, 10]
        return x


def build_model(num_classes=10):
    return VGGLikeCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = build_model(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)