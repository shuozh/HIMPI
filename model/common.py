import torch
import torch.nn as nn
import numpy as np



def activate(act='prelu'):
    if act == 'prelu':
        return nn.PReLU()
    if act == 'relu':
        return nn.ReLU(inplace=True)


class Spacial2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, if_bn=False, bias=True, act='prelu'):
        super(Spacial2D, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        m = []
        m.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

    def forward(self, x):
        # batch, channel, height, width = x.shape
        x = self.spaconv(x)
        return x


class Spacial3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, if_bn=False, bias=True,
                 act='prelu'):
        super(Spacial3D, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        m = []
        m.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm3d(out_channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

    def forward(self, x):
        # batch, channel, height, width = x.shape
        x = self.spaconv(x)
        return x



class ResidualDownSample(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(ResidualDownSample, self).__init__()

        m = []
        m.append(nn.Conv2d(channels, 2 * channels, 3, 2, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(2 * channels))
        if act is not None:
            m.append(activate(act))


        self.top = nn.Sequential(*m)


    def forward(self, x):
        top = self.top(x)
        return top


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels, if_bn, bias, act))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ResidualUpSample(nn.Module):
    def __init__(self, channels, if_bn=False, bias=False, act='prelu'):
        super(ResidualUpSample, self).__init__()

        m = []
        m.append(nn.ConvTranspose2d(channels, channels // 2, 3, 2, 1, output_padding=1,
                                    bias=bias), )
        if if_bn:
            m.append(nn.BatchNorm2d(channels // 2))
        if act is not None:
            m.append(activate(act))

        self.top = nn.Sequential(*m)



    def forward(self, x):
        top = self.top(x)
        return top


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels, if_bn, bias, act))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class DownSample4D(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(DownSample4D, self).__init__()
        self.down = DownSample(in_channels, scale_factor, stride, if_bn, bias, act)
        self.scale_factor = scale_factor

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.down(x)
        x = x.reshape(batch, height_view, width_view, channel * self.scale_factor, int(height // self.scale_factor),
                      int(width // self.scale_factor))
        x = x.permute(0, 3, 1, 2, 4, 5)
        return x


class UpSample4D(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, if_bn=False, bias=True, act='prelu'):
        super(UpSample4D, self).__init__()
        self.up = UpSample(in_channels, scale_factor, stride, if_bn, bias, act)
        self.scale_factor = scale_factor

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.up(x)
        x = x.reshape(batch, height_view, width_view, int(channel // self.scale_factor), height * self.scale_factor,
                      width * self.scale_factor)
        x = x.permute(0, 3, 1, 2, 4, 5)
        return x



class SAD(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(SAD, self).__init__()
        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 0, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.angconv = nn.Sequential(*m)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape

        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.spaconv(x)
        x = x.reshape(batch, height_view, width_view, channel, height, width)

        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(batch * height * width, channel, height_view, width_view)
        x = self.angconv(x)
        x = x.reshape(batch, height, width, channel, height_view-2, width_view-2)

        x = x.permute(0, 3, 4, 5, 1, 2)

        return x


class SA(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(SA, self).__init__()
        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.spaconv = nn.Sequential(*m)

        m = []
        m.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=bias))
        if if_bn:
            m.append(nn.BatchNorm2d(channels))
        if act is not None:
            m.append(activate(act))
        self.angconv = nn.Sequential(*m)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape

        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.spaconv(x)
        x = x.reshape(batch, height_view, width_view, channel, height, width)

        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(batch * height * width, channel, height_view, width_view)
        x = self.angconv(x)
        x = x.reshape(batch, height, width, channel, height_view, width_view)

        x = x.permute(0, 3, 4, 5, 1, 2)

        return x
    

class MultiSAD(nn.Module):
    def __init__(self, num=2, channels=64, if_bn=False, bias=True, act='prelu'):
        super(MultiSAD, self).__init__()
        m = []
        for i in range(num):
            m.append(SAD(channels, if_bn, bias, act))
        self.SAD = nn.Sequential(*m)

    def forward(self, x):
        x = self.SAD(x)
        return x


class MultiSA(nn.Module):
    def __init__(self, num=2, channels=64, if_bn=False, bias=True, act='prelu'):
        super(MultiSA, self).__init__()
        m = []
        for i in range(num):
            m.append(SA(channels, if_bn, bias, act))
        self.SA = nn.Sequential(*m)

    def forward(self, x):
        x = self.SA(x)
        return x



class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True, act='prelu'):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias),
            activate(act),
            nn.Linear(channel // reduction, channel, bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, height, width = x.shape
        y = self.avg_pool(x).reshape(batch, channel)
        # print(y.shape)
        y = self.mlp(y).reshape(batch, channel, 1, 1)
        return x * y


class MultiSpacial2D(nn.Module):
    def __init__(self, channels, num, if_bn=False, bias=True, act='prelu'):
        super(MultiSpacial2D, self).__init__()
        m = []
        for i in range(num):
            m.append(Spacial2D(channels, channels, if_bn=if_bn, bias=bias,  act=act))
        self.top = nn.Sequential(*m)

    def forward(self, x):
        x = self.top(x)
        return x


class MultiSpacial3D(nn.Module):
    def __init__(self, channels, num, if_bn=False, bias=True, act='prelu'):
        super(MultiSpacial3D, self).__init__()
        m = []
        for i in range(num):
            m.append(Spacial3D(channels, channels, if_bn=if_bn, bias=bias, act=act))
        self.top = nn.Sequential(*m)

    def forward(self, x):
        x = self.top(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(EncoderDecoder, self).__init__()
        # self.channels = channels
        m = []
        m.append(MultiSpacial2D(channels, 2, if_bn, bias, act))
        self.encoder_1_0 = nn.Sequential(*m)
        m = []
        m.append(DownSample(channels, 2, 2, if_bn, bias, act))
        self.encoder_1_1 = nn.Sequential(*m)

        m = []
        m.append(MultiSpacial2D(2 * channels, 2, if_bn, bias, act))
        self.encoder_2_0 = nn.Sequential(*m)
        m = []
        m.append(DownSample(2 * channels, 2, 2, if_bn, bias, act))
        self.encoder_2_1 = nn.Sequential(*m)

        m = []
        m.append(MultiSpacial2D(4 * channels, 2, if_bn, bias, act))
        self.encoder_3_0 = nn.Sequential(*m)
        m = []
        m.append(DownSample(4 * channels, 2, 2, if_bn, bias, act))
        self.encoder_3_1 = nn.Sequential(*m)

        self.bottom = Spacial2D(channels * 8, channels * 8, 3,1,1,if_bn, bias, act)

        self.decoder_3_0 = UpSample(channels * 8, 2, 2, if_bn, bias, act)
        m = []
        m.append(Spacial2D(8 * channels, 4 * channels,3,1,1, if_bn, bias, act))
        m.append(MultiSpacial2D(4 * channels, 2, if_bn, bias, act))
        self.decoder_3_1 = nn.Sequential(*m)

        self.decoder_2_0 = UpSample(channels * 4, 2, 2, if_bn, bias, act)
        m = []
        m.append(Spacial2D(4 * channels, 2 * channels, 3,1,1,if_bn, bias, act))
        m.append(MultiSpacial2D(2 * channels, 2, if_bn, bias, act))
        self.decoder_2_1 = nn.Sequential(*m)

        self.decoder_1_0 = UpSample(channels * 2, 2, 2, if_bn, bias, act)
        m = []
        m.append(Spacial2D(2 * channels, channels, 3,1,1,if_bn, bias, act))
        m.append(MultiSpacial2D(channels, 2, if_bn, bias, act))
        self.decoder_1_1 = nn.Sequential(*m)

    def forward(self, x):
        # batch, channel, height, width = x.shape
        encoder_1_res = self.encoder_1_0(x)
        encoder_1 = self.encoder_1_1(encoder_1_res)

        encoder_2_res = self.encoder_2_0(encoder_1)
        encoder_2 = self.encoder_2_1(encoder_2_res)

        encoder_3_res = self.encoder_3_0(encoder_2)
        encoder_3 = self.encoder_3_1(encoder_3_res)

        bottom = self.bottom(encoder_3)

        decoder_3 = self.decoder_3_0(bottom)
        decoder_3 = torch.cat((decoder_3, encoder_3_res), dim=1)
        decoder_3 = self.decoder_3_1(decoder_3)

        decoder_2 = self.decoder_2_0(decoder_3)
        decoder_2 = torch.cat((decoder_2, encoder_2_res), dim=1)
        decoder_2 = self.decoder_2_1(decoder_2)

        decoder_1 = self.decoder_1_0(decoder_2)
        decoder_1 = torch.cat((decoder_1, encoder_1_res), dim=1)
        decoder_1 = self.decoder_1_1(decoder_1)

        return decoder_1




class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,3,1,1)#conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes,planes,3,1,1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MRTR(nn.Module):
    def __init__(self, channel):
        super(MRTR, self).__init__()
        self.channel = channel
        self.conv_dila1 = nn.Conv2d(channel,channel,3,1,1)
        self.conv_dila2 = nn.Conv2d(channel,channel,3,1,2,2)
        self.conv1 = nn.Conv2d(2*channel,channel,3,1,1)
        self.conv2 = nn.Conv2d(channel,channel,3,1,1)
        self.conv3 = nn.Conv2d(channel,channel,3,1,1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x1,x2):
        x2_1 = self.conv_dila1(x2)
        x2_2 = self.conv_dila2(x2)
        x2 = self.conv1(torch.cat((x2_1,x2_2),1))
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = x1 - x2
        x2 = (1-self.sigmod(x2))
        x1 = x1 * x2
        return x1