import torch
import torch.nn as nn

class Anisotrpic_MultiDiscriminator3D(nn.Module):
    def __init__(self, channels=1, num_scales=3, num_filters=64):
        super(Anisotrpic_MultiDiscriminator3D, self).__init__()

        def anisotropic_discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=[1, 4, 4], stride=[1, 2, 2], padding=[0, 1, 1])]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, kernel_size=[3, 4, 4], stride=[1, 2, 2], padding=[1, 1, 1])]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(num_scales):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, num_filters, normalize=False),
                    *discriminator_block(num_filters, num_filters * 2),
                    *discriminator_block(num_filters * 2, num_filters * 4),
                    *discriminator_block(num_filters * 4, num_filters * 8),
                    nn.Conv3d(num_filters * 8, 1, [1, 3, 3], padding=[0, 1, 1])
                ),
            )

        self.downsample = nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)  # Downsample resolution
        return outputs


if __name__=="__main__":
    model = Anisotrpic_MultiDiscriminator3D(channels=1, num_scales=2, num_filters=64)
    input_data = torch.randn(1, 1, 16, 256, 256)
    output = model(input_data)
    print(output[0].shape, output[1].shape)

    from torchinfo import summary
    summary(model, (1, 1, 16, 256, 256), device='cpu')