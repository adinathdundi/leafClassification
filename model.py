"""
model.py

PyTorch port of the well-known Keras "CIFAR-10 ResNet v1" example
(resnet_layer / resnet_v1), same structure and hyperparameters:

    - depth = 6n + 2  ->  ResNet20 (n=3), ResNet32 (n=5), ResNet44 (n=7),
                          ResNet56 (n=9), ResNet110 (n=18), etc.
    - 3 stacks of residual blocks, filters starting at 16 and doubling
      at the first block of each new stack (where spatial resolution
      is also halved via stride=2)
    - He-normal weight init, batch norm, ReLU
    - identity shortcut where shapes match; a 1x1 conv (no BN, no
      activation) projection shortcut where downsampling/channel-change
      happens

NOTE ON NAMING: this is NOT the same family as the standard ImageNet
"ResNet-18" (which has a 7x7 stride-2 stem + maxpool, and 4 stages of
[64,128,256,512] filters with 2 blocks each). This is the CIFAR-style
ResNet-v1 family, parameterized by depth=6n+2. Pick whichever `depth`
you want (e.g. 20 is a fast, well-tested starting point); "ResNet18"
was the caller's shorthand but doesn't correspond to a specific depth
in *this* family.

ADAPTATION FROM THE ORIGINAL: the original ends with
`AveragePooling2D(pool_size=8)`, which only works because CIFAR images
are a fixed 32x32 (-> 8x8 feature maps after 2 stride-2 downsamples).
Since our leaf images are resized to whatever `img_size` you choose
(224 by default), we use `AdaptiveAvgPool2d(1)` instead, which pools
to 1x1 regardless of input resolution — functionally the same idea
(global average pooling before the classifier head), just resolution-
independent.

kernel_regularizer=l2(1e-4) in the original has no direct per-layer
equivalent in PyTorch; the closest match is `weight_decay=1e-4` on the
optimizer in train.py, applied to all parameters (Keras' l2() applied
only to conv kernels, not biases/BN — a minor difference worth knowing
about, not usually consequential in practice).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    """PyTorch equivalent of the Keras `resnet_layer` helper: a single
    Conv2D (+ optional BatchNorm + optional activation), in either
    conv-first or BN/activation-first ("pre-activation") order.
    """

    def __init__(self, in_channels, num_filters=16, kernel_size=3, stride=1,
                 activation="relu", batch_normalization=True, conv_first=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, num_filters, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=True,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

        self.bn = nn.BatchNorm2d(num_filters) if batch_normalization else None
        self.activation = activation
        self.conv_first = conv_first

    def _act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        return x  # activation=None -> identity, matching the Keras version

    def forward(self, x):
        if self.conv_first:
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            x = self._act(x)
        else:
            if self.bn is not None:
                x = self.bn(x)
            x = self._act(x)
            x = self.conv(x)
        return x


class BasicResBlock(nn.Module):
    """One residual block: two 3x3 ResNetLayers, plus a shortcut that's
    either the identity (shapes already match) or a 1x1 projection conv
    (no BN, no activation — matching the Keras branch that only fires
    "if stack > 0 and res_block == 0").
    """

    def __init__(self, in_channels, num_filters, stride, downsample):
        super().__init__()
        self.conv1 = ResNetLayer(in_channels, num_filters, kernel_size=3,
                                  stride=stride, activation="relu",
                                  batch_normalization=True, conv_first=True)
        self.conv2 = ResNetLayer(num_filters, num_filters, kernel_size=3,
                                  stride=1, activation=None,
                                  batch_normalization=True, conv_first=True)
        if downsample:
            self.shortcut = ResNetLayer(in_channels, num_filters, kernel_size=1,
                                         stride=stride, activation=None,
                                         batch_normalization=False, conv_first=True)
        else:
            self.shortcut = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(shortcut + y)
        return out


class ResNetV1(nn.Module):
    """Direct port of the Keras `resnet_v1(input_shape, depth, num_classes)`.

    Parameters
    ----------
    in_channels : int
        3 for RGB mode, 1 for silhouette-mask mode.
    depth : int
        Must satisfy depth = 6n + 2 (e.g. 20, 32, 44, 56, 110).
    num_classes : int
    """

    def __init__(self, in_channels=3, depth=20, num_classes=38):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(
                f"depth should be 6n+2 (e.g. 20, 32, 44, 56, 110); got {depth}"
            )
        num_filters = 16
        num_res_blocks = (depth - 2) // 6

        # stem: resnet_layer(inputs=inputs) with all defaults
        self.stem = ResNetLayer(in_channels, num_filters)

        blocks = []
        current_channels = num_filters
        for stack in range(3):
            for res_block in range(num_res_blocks):
                stride = 1
                downsample = False
                if stack > 0 and res_block == 0:
                    stride = 2       # Downsample, matching the Keras comment
                    downsample = True
                blocks.append(BasicResBlock(current_channels, num_filters, stride, downsample))
                current_channels = num_filters
            num_filters *= 2  # doubles AFTER each full stack, same as the Keras loop
        self.blocks = nn.ModuleList(blocks)

        # AveragePooling2D(pool_size=8) in the original assumes fixed 32x32 CIFAR
        # input; AdaptiveAvgPool2d(1) does the same "global average pool before
        # the classifier" job regardless of input resolution.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(current_channels, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # NOTE: returns raw logits, not softmax probabilities. The original
        # Keras model ends in Dense(..., activation='softmax'); in PyTorch the
        # idiomatic equivalent is to return logits and use nn.CrossEntropyLoss
        # (which applies log-softmax internally) -- that's what train.py does.
        # Softmax-ing here AND using CrossEntropyLoss would double-apply softmax.
        return x


if __name__ == "__main__":
    # quick shape + parameter-count sanity check across a few depths and
    # both input-channel configs (rgb=3, mask=1)
    for depth in (20, 32):
        for c in (3, 1):
            m = ResNetV1(in_channels=c, depth=depth, num_classes=38)
            n_params = sum(p.numel() for p in m.parameters())
            dummy = torch.randn(2, c, 224, 224)
            out = m(dummy)
            print(f"depth={depth} in_channels={c} -> output shape {tuple(out.shape)}, "
                  f"params={n_params:,}")
