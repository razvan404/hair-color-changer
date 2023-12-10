import torch
import torch.nn as nn


def _Conv2dExt(in_num_channels: int, out_num_channels: int, is_transpose: bool = False):
    batch_norm = nn.BatchNorm2d(out_num_channels)
    if is_transpose:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_num_channels, out_num_channels, kernel_size=2, stride=2
            ),
            batch_norm,
        )
    conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size=3, padding=1)
    activation = nn.ReLU()
    return nn.Sequential(conv, batch_norm, activation)


class TwoConvDownSampling(nn.Module):
    def __init__(
        self, in_num_channels: int, hid_num_channels: int, out_num_channels: int
    ):
        super(TwoConvDownSampling, self).__init__()
        self.conv1 = _Conv2dExt(in_num_channels, hid_num_channels)
        self.conv2 = _Conv2dExt(hid_num_channels, out_num_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x_resized = self.maxpool(x)
        return x_resized, x


def ThreeConvUpSampling(
    in_num_channels: int, hid_num_channels: int, out_num_channels: int
):
    return nn.Sequential(
        _Conv2dExt(in_num_channels, hid_num_channels),
        _Conv2dExt(hid_num_channels, hid_num_channels),
        _Conv2dExt(hid_num_channels, out_num_channels, is_transpose=True),
    )


class SegUNet(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super(SegUNet, self).__init__()
        self.downscale_layers = [
            TwoConvDownSampling(num_channels, 64, 64),
            TwoConvDownSampling(64, 128, 128),
            TwoConvDownSampling(128, 256, 256),
            TwoConvDownSampling(256, 512, 512),
        ]
        self.upscale_layers = [
            ThreeConvUpSampling(512, 1024, 512),
            ThreeConvUpSampling(1024, 512, 256),
            ThreeConvUpSampling(512, 256, 128),
            ThreeConvUpSampling(256, 128, 64),
        ]
        self.last_conv1 = _Conv2dExt(128, 64)
        self.last_conv2 = _Conv2dExt(64, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        prev_xs = []
        for layer in self.downscale_layers:
            x, unscaled_x = layer(x)
            prev_xs.append(unscaled_x)
        x = self.upscale_layers[0](x)
        for prev_x, layer in zip(reversed(prev_xs[1:]), self.upscale_layers[1:]):
            x = layer(torch.cat((prev_x, x), dim=1))
        x = self.last_conv1(torch.cat((prev_xs[0], x), dim=1))
        x = self.last_conv2(x)
        x = self.out_conv(x)
        return x


def test_down_sampling():
    with torch.no_grad():
        img_size = 128
        num_channels = 3
        out_channels = 64
        batch_size = 16
        x = torch.rand((batch_size, num_channels, img_size, img_size))
        model = TwoConvDownSampling(num_channels, 64, out_channels)
        x, x1 = model(x)
        assert x1.shape == (batch_size, out_channels, img_size, img_size)
        assert x.shape == (batch_size, out_channels, img_size // 2, img_size // 2)


def test_up_sampling():
    with torch.no_grad():
        img_size = 64
        num_channels = 3
        out_channels = 64
        batch_size = 16
        x = torch.rand((batch_size, num_channels, img_size, img_size))
        model = ThreeConvUpSampling(num_channels, 64, out_channels)
        x = model(x)
        assert x.shape == (batch_size, out_channels, img_size * 2, img_size * 2)


def test_seg_unet():
    with torch.no_grad():
        img_size = 240
        num_channels = 3
        num_classes = 10
        batch_size = 16
        x = torch.rand((batch_size, num_channels, img_size, img_size))
        model = SegUNet(num_channels, num_classes)
        x = model(x)
        assert x.shape == (batch_size, num_classes, img_size, img_size)


if __name__ == "__main__":
    test_down_sampling()
    test_up_sampling()
    test_seg_unet()
