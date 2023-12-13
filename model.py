import torch
import torch.nn as nn
import torchvision.transforms.functional as F


def TwoConv2d(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SegmentationUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: list[int] = [64, 128, 256, 512],
    ):
        super(SegmentationUNet, self).__init__()
        self.down_sampling = nn.ModuleList()
        for feature in features:
            self.down_sampling.append(TwoConv2d(in_channels, feature))
            in_channels = feature

        self.max_pool = nn.MaxPool2d(2, 2)
        self.bottleneck = TwoConv2d(features[-1], features[-1] * 2)

        self.up_sampling = nn.ModuleList()
        for feature in reversed(features):
            self.up_sampling.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.up_sampling.append(TwoConv2d(feature * 2, feature))

        self.last_conv = nn.Conv2d(features[0], out_channels, 1)
        self.num_features = len(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for layer in self.down_sampling:
            x = layer(x)
            skip_connections.append(x)
            x = self.max_pool(x)
        skip_connections.reverse()
        x = self.bottleneck(x)
        for i in range(self.num_features):
            x = self.up_sampling[2 * i](x)
            if x.shape != skip_connections[i].shape:
                F.resize(x, size=skip_connections[i].shape[2:])
            x = torch.cat((skip_connections[i], x), dim=1)
            x = self.up_sampling[2 * i + 1](x)
        x = self.last_conv(x)
        return x


def test_segmentation_unet():
    with torch.no_grad():
        img_size = 240
        num_channels = 3
        num_classes = 10
        batch_size = 16
        x = torch.rand((batch_size, num_channels, img_size, img_size))
        model = SegmentationUNet(num_channels, num_classes)
        x = model(x)
        assert x.shape == (batch_size, num_classes, img_size, img_size)


if __name__ == "__main__":
    test_segmentation_unet()
