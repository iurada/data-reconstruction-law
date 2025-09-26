import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks=4, width=128, stem_kernel=3):
        super().__init__()
        self.num_blocks = num_blocks

        self.stem = nn.Conv2d(in_channels, width, kernel_size=stem_kernel, padding=1, bias=False)
        self._init_stem_like_large(self.stem.weight)

        # Residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(_ResidualBlock(width, width, init_fn=self._init_tied))
        self.blocks = nn.Sequential(*blocks)

        # Head setup (determine feature size)
        with torch.no_grad():
            dummy = torch.randn((1, in_channels, 32, 32))
            out = self.stem(dummy)
            out = F.relu(out)
            out = self.blocks(out)
            head_size = out.view(out.size(0), -1).size(1)

            tot_params = self.stem.weight.numel()
            for block in self.blocks:
                tot_params += block.conv1.weight.numel()
                tot_params += block.conv2.weight.numel()
                if block.proj is not None:
                    tot_params += block.proj.weight.numel()
            tot_params += head_size
            self.pL = head_size
            print(f'p: {tot_params} | pL: {head_size}')

        # Head linear
        self.head = nn.Linear(head_size, num_classes, bias=False)
        self._init_head_tied(self.head.weight)

    def _init_stem_like_large(self, w):
        fan_in = w.size(1) * (32*32)
        if w.dim() > 2 and w.size(0) % 2 == 0:
            w.data.copy_(torch.randn((w.size(0),) + w.shape[1:], device=w.device) / math.sqrt(fan_in))
        else:
            nn.init.normal_(w, mean=0.0, std=1.0 / math.sqrt(fan_in))

    def _init_tied(self, w):
        fan_in = w.size(1) * (w[0][0].numel() if w.dim() > 2 else 1)
        if w.dim() > 2 and w.size(0) % 2 == 0:
            w.data.copy_(torch.randn((w.size(0),) + w.shape[1:], device=w.device) / math.sqrt(fan_in))
        else:
            nn.init.normal_(w, mean=0.0, std=1.0 / math.sqrt(fan_in))

    def _init_head_tied(self, w):
        _, width = w.shape
        nn.init.normal_(w, mean=0.0, std=(1.0 / math.sqrt(width)))

    def forward(self, x):
        out = self.stem(x)
        out = F.relu(out)
        out = self.blocks(out)
        out = out.view(out.size(0), -1)
        return self.head(out)


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, init_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        init_fn(self.conv1.weight)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        init_fn(self.conv2.weight)

        # optional projection
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_ch)
            init_fn(self.proj.weight)
        else:
            self.proj = None

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)

        if self.proj is not None:
            identity = self.proj(identity)
            identity = self.bn_proj(identity)

        out = identity + out
        return self.activation(out)