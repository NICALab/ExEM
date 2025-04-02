"""
Anisotropic 3D U-Net model for VCM application.
Variant of the following paper.
Lee, Kisuk, et al. "Superhuman accuracy on the SNEMI3D connectomics challenge." arXiv preprint arXiv:1706.00120 (2017).
"""

import torch
import torch.nn as nn

def make_group_norm(num_channels):
    if num_channels in [1, 2, 4, 8]:
        suggested_num_groups = 1
    elif num_channels in [16]:
        suggested_num_groups = 4
    elif num_channels in [32, 64]:
        suggested_num_groups = 8
    elif num_channels in [128]:
        suggested_num_groups = 16
    elif num_channels in [256, 512]:
        suggested_num_groups = 32
    else:
        raise ValueError("Invalid number of channels.")
    
    return nn.GroupNorm(suggested_num_groups, num_channels)


class Anisotropic_TripleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.elu = nn.ELU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.groupnorm1 = make_group_norm(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.groupnorm2 = make_group_norm(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.groupnorm3 = make_group_norm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.groupnorm1(x)
        x1 = self.elu(x)

        x = self.conv2(x1)
        x = self.groupnorm2(x)
        x = self.elu(x)

        x = self.conv3(x)
        x = self.groupnorm3(x)
        x = self.elu(x)

        return x1 + x


class Anisotrpic_UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, num_channels=[28, 36, 48, 64, 80]):
        super(Anisotrpic_UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.downsample = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        
        self.elu = nn.ELU(inplace=True)
        self.first_conv = nn.Conv3d(n_channels, num_channels[0], kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=True)

        self.tripleconv1 = Anisotropic_TripleConv3D(num_channels[0], num_channels[0])
        self.tripleconv2 = Anisotropic_TripleConv3D(num_channels[0], num_channels[1])
        self.tripleconv3 = Anisotropic_TripleConv3D(num_channels[1], num_channels[2])
        self.tripleconv4 = Anisotropic_TripleConv3D(num_channels[2], num_channels[3])
        self.tripleconv5 = Anisotropic_TripleConv3D(num_channels[3], num_channels[4])

        self.tripleconv_up1 = Anisotropic_TripleConv3D(num_channels[4] + num_channels[3], num_channels[3])
        self.tripleconv_up2 = Anisotropic_TripleConv3D(num_channels[3] + num_channels[2], num_channels[2])
        self.tripleconv_up3 = Anisotropic_TripleConv3D(num_channels[2] + num_channels[1], num_channels[1])
        self.tripleconv_up4 = Anisotropic_TripleConv3D(num_channels[1] + num_channels[0], num_channels[0])

        self.final_conv = nn.Conv3d(num_channels[0], num_channels[0], kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=True)
        self.one_by_one_conv = nn.Conv3d(num_channels[0], n_classes, kernel_size=1, bias=True)


    def forward(self, x):
        x = self.first_conv(x)
        x = self.elu(x)

        x1 = self.tripleconv1(x)
        x = self.downsample(x1)
        x2 = self.tripleconv2(x)
        x = self.downsample(x2)
        x3 = self.tripleconv3(x)
        x = self.downsample(x3)
        x4 = self.tripleconv4(x)
        x = self.downsample(x4)
        x = self.tripleconv5(x)
        x = self.upsample(x)

        x = torch.cat([x, x4], dim=1)
        x = self.tripleconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.tripleconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.tripleconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.tripleconv_up4(x)
        x = self.final_conv(x)
        x = self.elu(x)
        x = self.one_by_one_conv(x)

        return x


if __name__=="__main__":
    from torchinfo import summary
    model = Anisotrpic_UNet3D(1, 1, num_channels=[32, 64, 128, 256, 512])
    input_data = torch.randn(1, 1, 16, 256, 256)
    output_data = model(input_data)
    print(output_data.shape)

    summary(model, input_size=(1, 1, 16, 256, 256), device='cpu')