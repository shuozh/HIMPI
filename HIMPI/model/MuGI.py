import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class Dconv_Feature(nn.Module):
    def __init__(self, channel,if_mlp = False,mlp_layers= 11):
        super(Dconv_Feature,self).__init__()
        if if_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_layers,2*mlp_layers),
                nn.ReLU(),
                nn.Linear(2*mlp_layers,mlp_layers),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(channel,channel,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv1d(channel,channel,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
            )
        self.if_mlp = if_mlp
    def forward(self,x):
        b,l,c,h,w = list(x.shape)
        if self.if_mlp:
            x = rearrange(x,'b l c h w -> (b h w c) l')
            x = self.mlp(x)
            x = rearrange(x,'(b h w c) l -> (b l) c h w',b=b,l=l,c=c,h=h,w=w)
        else:
            x = rearrange(x,'b l c h w -> (b h w) c l')
            x = self.conv(x)
            x = rearrange(x,'(b h w) c l -> (b l) c h w',b=b,l=l,c=c,h=h,w=w)
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)





class MuGIBlocks1(nn.Module):
    def __init__(self, c,l = 11, shared_b=False):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.df1 = Dconv_Feature(c,if_mlp=True,mlp_layers=l)
        self.df2 = Dconv_Feature(c,if_mlp=True,mlp_layers=l)
        self.df3 = Dconv_Feature(c,if_mlp=True,mlp_layers=l)
        self.df4 = Dconv_Feature(c,if_mlp=True,mlp_layers=l)
        self.l = l
    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x = self.df1(rearrange(x,'(b l) c h w -> b l c h w',l=self.l))
        y = self.df2(rearrange(y,'(b l) c h w -> b l c h w',l=self.l))
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        x = self.df3(rearrange(x,'(b l) c h w -> b l c h w',l=self.l))
        y = self.df4(rearrange(y,'(b l) c h w -> b l c h w',l=self.l))
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r