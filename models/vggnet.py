import torch
from torch import nn
from d2l import torch as d2l

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # VGG-11


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vggNet(vgg_block_param):
    conv_blks = []
    in_channels = 5
    # The convolutional part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 3)
    )


# net = vggNet(vgg_block_param=conv_arch)
#
# X = torch.randn(size=(1, 5, 240, 240))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)
