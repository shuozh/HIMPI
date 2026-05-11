import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return x_LL, x_HL, x_LH, x_HH

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1) # [N,2,H,W] could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class HWABS1(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
        super(HWABS1, self).__init__()
        self.dwt1 = DWT()
        self.iwt1 = IWT()
        self.dwt2 = DWT()
        self.iwt2 = IWT()

        modules_body = \
            [
                conv(int(n_feat*2.5), n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat*2, kernel_size, bias=bias)
            ]
        self.body1 = nn.Sequential(*modules_body)
        self.body2 = nn.Sequential(*modules_body)

        self.WSA1 = SALayer()
        self.WCA1 = CALayer(n_feat*2, reduction, bias=bias)
        self.WSA2 = SALayer()
        self.WCA2 = CALayer(n_feat*2, reduction, bias=bias)

        self.conv1x1_1 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)
        self.conv3x3_1 = nn.Conv2d(int(n_feat), o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate_1 = act
        self.conv1x1_final_1 = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)
        self.conv1x1_2 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)
        self.conv3x3_2 = nn.Conv2d(int(n_feat), o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate_2 = act
        self.conv1x1_final_2 = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x1, x2):
        residual1 = x1
        residual2 = x2

        # Split 2 part
        wavelet_path_in1, identity_path1 = torch.chunk(x1, 2, dim=1)
        wavelet_path_in2, identity_path2 = torch.chunk(x2, 2, dim=1)

        # Wavelet domain (Dual attention)
        x1_LL, x1_HL, x1_LH, x1_HH = self.dwt1(wavelet_path_in1)
        x2_LL, x2_HL, x2_LH, x2_HH = self.dwt2(wavelet_path_in2)

        x_dwt1 = torch.cat([x1_LL, x1_HL, x1_LH, x1_HH], dim=1)
        res1 = self.body1(torch.cat([x2_LL, x_dwt1], dim=1))
        branch_sa1 = self.WSA1(res1)
        branch_ca1 = self.WCA1(res1)
        res1 = torch.cat([branch_sa1, branch_ca1], dim=1)
        res1 = self.conv1x1_1(res1) + x_dwt1
        wavelet_path1 = self.iwt1(res1)


        x_dwt2 = torch.cat([x2_LL, x2_HL, x2_LH,x2_HH], dim=1)
        res2 = self.body2(torch.cat([x1_LL, x_dwt2], dim=1))
        branch_sa2 = self.WSA2(res2)
        branch_ca2 = self.WCA2(res2)
        res2 = torch.cat([branch_sa2, branch_ca2], dim=1)
        res2 = self.conv1x1_2(res2) + x_dwt2
        wavelet_path2 = self.iwt2(res2)

        out1 = torch.cat([wavelet_path1, identity_path1], dim=1)
        out1 = self.activate_1(self.conv3x3_1(out1))
        out1 += self.conv1x1_final_1(residual1)
        out2 = torch.cat([wavelet_path2, identity_path2], dim=1)
        out2 = self.activate_2(self.conv3x3_2(out2))
        out2 += self.conv1x1_final_2(residual2)

        return out1, out2

