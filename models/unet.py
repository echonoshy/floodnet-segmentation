import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A convolutional block consisting of two convolution layers each followed by 
    a normalization layer and a ReLU activation.
    """
    def __init__(self, ch_in, ch_out, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            norm_layer(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    """
    An upsampling block using nn.Upsample and a convolutional layer,
    followed by a normalization layer and ReLU activation.
    """
    def __init__(self, ch_in, ch_out, norm_layer=None):
        super(UpConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            norm_layer(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    """
    def __init__(self, img_ch=3, output_ch=1, norm_layer=None):
        super(UNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64, norm_layer=norm_layer)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128, norm_layer=norm_layer)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256, norm_layer=norm_layer)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512, norm_layer=norm_layer)
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024, norm_layer=norm_layer)

        self.Up5 = UpConv(ch_in=1024, ch_out=512, norm_layer=norm_layer)
        self.Up_conv5 = ConvBlock(ch_in=1024, ch_out=512, norm_layer=norm_layer)

        self.Up4 = UpConv(ch_in=512, ch_out=256, norm_layer=norm_layer)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256, norm_layer=norm_layer)

        self.Up3 = UpConv(ch_in=256, ch_out=128, norm_layer=norm_layer)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128, norm_layer=norm_layer)

        self.Up2 = UpConv(ch_in=128, ch_out=64, norm_layer=norm_layer)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64, norm_layer=norm_layer)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoding + Concatenation path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)

