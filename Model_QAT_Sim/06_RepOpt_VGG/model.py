# model.py
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RepOptVGGLike(nn.Module):
    """
    INT8/NPU-friendly plain VGG-style CNN for CIFAR-10

    Input : [B, 3, 32, 32]
    Output: [B, 10]
    """

    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()

        c64 = int(64 * width_mult)
        c128 = int(128 * width_mult)
        c256 = int(256 * width_mult)
        c512 = int(512 * width_mult)

        # 32x32 -> 16x16
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, c64),
            ConvBNReLU(c64, c64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 16x16 -> 8x8
        self.stage2 = nn.Sequential(
            ConvBNReLU(c64, c128),
            ConvBNReLU(c128, c128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 8x8 -> 4x4
        self.stage3 = nn.Sequential(
            ConvBNReLU(c128, c256),
            ConvBNReLU(c256, c256),
            ConvBNReLU(c256, c256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # keep 4x4
        self.stage4 = nn.Sequential(
            ConvBNReLU(c256, c512),
            ConvBNReLU(c512, c512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stage1(x)   # [B, c64, 16, 16]
        x = self.stage2(x)   # [B, c128, 8, 8]
        x = self.stage3(x)   # [B, c256, 4, 4]
        x = self.stage4(x)   # [B, c512, 4, 4]
        x = self.avgpool(x)  # [B, c512, 1, 1]
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(num_classes=10, width_mult=1.0):
    return RepOptVGGLike(num_classes=num_classes, width_mult=width_mult)


if __name__ == "__main__":
    model = build_model(num_classes=10, width_mult=1.0)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(model)
    print("Output shape:", y.shape)