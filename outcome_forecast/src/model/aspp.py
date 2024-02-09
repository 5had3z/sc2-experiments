import torch
from torch import nn, Tensor
from torch.nn import functional as F
from konductor.models import MODEL_REGISTRY


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        modules = [
            nn.Conv2d(
                in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(),
        )

    def set_image_pooling(self, pool_size: int | None = None):
        self.aspp_pooling[0] = (
            nn.AdaptiveAvgPool2d(1)
            if pool_size is None
            else nn.AvgPool2d(kernel_size=pool_size, stride=1)
        )

    def forward(self, x: Tensor):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


@MODEL_REGISTRY.register_module("aspp")
class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: list[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )
        )

        for rate in rates:
            self.convs.append(ASPPConv(in_ch, out_ch, rate))
        self.convs.append(ASPPPooling(in_ch, out_ch))

        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor):
        res = torch.cat([conv(x) for conv in self.convs], dim=1)
        proj = self.project(res)
        return proj


# class ASPPDecoder(nn.Module):
#     def __init__(self, in_ch: int, hidden_ch:  int, rates:  list[int]):
