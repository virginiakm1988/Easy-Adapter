import torch
from torch import nn


class HoulsbyAdapter(nn.Module):
    """Implementation of Houlsby's Adapter
    References: https://arxiv.org/abs/1902.00751.
    """

    def __init__(self, input_size, bottleneck=128):
        super().__init__()

        self.houlsby_adapter = nn.Sequential(
            nn.Linear(input_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, input_size),
        )

    def forward(self, x):
        return self.houlsby_adapter(x)


class ConvAdapter(nn.Module):
    """Implementation of Convolution Adapter
    References: https://arxiv.org/abs/2301.07851.
    """

    def __init__(self, input_size, compress_rate=8, k=1, stride=1, dropout=0.8):
        super().__init__()

        def depthwise_conv(n_in, n_out, compress_rate, k, stride):
            conv = nn.Conv1d(n_in, n_out // compress_rate, k, stride=stride)
            nn.init.kaiming_normal_(conv.weight)
            return conv

        def pointwise_conv(n_in, n_out, compress_rate, k, stride):
            conv = nn.Conv1d(n_out // compress_rate, n_out, 1)
            nn.init.kaiming_normal_(conv.weight)
            return conv

        self.conv_adapter = nn.Sequential(
            depthwise_conv(input_size, input_size, compress_rate, k, stride),
            pointwise_conv(input_size, input_size, compress_rate, k, stride),
            nn.Dropout(p=dropout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv_adapter(x)


class AdapterBias(nn.Module):
    """Implementation of Adapter with Bias Vector
    References: https://arxiv.org/abs/2205.00305.
    """

    def __init__(self, input_size, dropout=0.8):
        super().__init__()
        self.adapter_vector = nn.Parameter(torch.ones((input_size), requires_grad=True))
        self.adapter_alpha = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.adapter_vector * self.adapter_alpha(x)

class LoRA(nn.Module):
    def __init__(
            self,
            input_size,
            dropout = 0.8,
            r = 16
        ):
        super().__init__()
        self.lora_adapter = lora.Linear(input_size, input_size, r)
        
    def forward(self, x):
        return self.lora_adapter(x)