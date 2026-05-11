import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.common import DownSample4D, UpSample4D,MultiSAD,MultiSA, Spacial2D, ca_layer,activate,UpSample
from model.ENN import ECNN,ICNN
from model.MuGI import MuGIBlocks1
from model.dwt1 import HWABS1
# class pre_part(nn.Module):
#     def __init__(self, channel):
#         super(pre_part, self).__init__()
#         self.conv = nn.Conv2d(3, channel//2, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv2 = nn.Conv2d(channel//2, channel, kernel_size=3, padding=1, stride=1, bias=True)
#         self.PRelu = nn.PReLU()
#         self.channel = channel

#     def forward(self, x):
#         batch, _, height_view, width_view, height, width = list(x.shape)
#         x = rearrange(x, 'b c u v x y -> (b u v) c x y')
#         x = self.conv(x)
#         x = self.PRelu(self.conv2(x))
#         x = rearrange(x, '(b u v) c x y -> b c u v x y', b=batch, c=self.channel, u=height_view, v=width_view)
#         return x

class pre_part(nn.Module):
    def __init__(self, channel):
        super(pre_part,self).__init__()
        self.conv = nn.Conv2d(3,channel//2, kernel_size=3, padding=1, stride=1)
        self.conv1d = nn.Conv1d(channel//2,channel//2,kernel_size =3 ,padding = 1,stride = 1)
        self.conv2 = nn.Conv2d(channel//2,channel,kernel_size=3,padding=1,stride=1)
        self.conv1d1 = nn.Conv1d(channel,channel,kernel_size =3 ,padding = 1,stride = 1)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.channel = channel
    
    def forward(self,x):
        b, l, u, v, c, h, w =list(x.shape)
        x = rearrange(x,'b l u v c h w -> (b l u v) c h w')
        x = self.conv(x)
        x = rearrange(x,'(b l u v) c h w ->(b u v h w) c l',b = b,l = l,u=u,v=v,h=h,w=w)
        x = self.conv1d(x)
        x = self.relu(x)
        x = rearrange(x,'(b u v h w) c l->(b l u v) c h w',b = b,l = l,u=u,v=v,h=h,w=w)
        x = self.conv2(x)
        x = rearrange(x,'(b l u v) c h w ->(b u v h w) c l',b = b,l = l,u=u,v=v,h=h,w=w)
        x = self.conv1d1(x)
        x = self.relu2(x)
        x = rearrange(x,'(b u v h w) c l->(b l) c u v h w',b = b,l = l,u=u,v=v,h=h,w=w)
        #x = rearrange(x,'(b l u v) c h w -> b l u v h w c',b=b,l=l,u=u,v=v)
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


class MultiSAS(nn.Module):
    def __init__(self, channels, if_bn=False, bias=True, act='prelu'):
        super(MultiSAS, self).__init__()

        self.down_0_1 = DownSample4D(channels, 2, 2, if_bn, bias, act)

        self.up_1 = UpSample(2 * channels, 2, 2, if_bn, bias, act)
        self.up_2 = UpSample(2 * channels, 2, 2, if_bn, bias, act)
        self.sad = SAD(channels, if_bn, bias, act)
        self.sa1 = SA(channels, if_bn, bias, act)
        self.sa2 = SA(channels, if_bn, bias, act)
        self.sa3 = SA(2*channels, if_bn, bias, act)
        self.sa4 = SA(2*channels, if_bn, bias, act)
        self.ca1 = ca_layer(2*channels, 8, bias, act)
        self.down1 = Spacial2D( 2*channels, channels, if_bn=if_bn, bias=bias, act=act)
        self.ca2 = ca_layer(2*channels, 8, bias, act)
        self.down2 = Spacial2D( 2*channels, channels, if_bn=if_bn, bias=bias, act=act)
    def forward(self, x):
        batch, channel, height_view, width_view, height, width = x.shape
        x = self.sad(x)
        x1 = self.sa1(x)
        x2 = self.sa2(x)

        xd = self.down_0_1(x)
        del x
        xd1 = self.sa3(xd)
        xd2 = self.sa4(xd)
        del xd
        xd1 = rearrange(xd1, 'b c u v x y -> (b u v) c x y')
        xd2 = rearrange(xd2, 'b c u v x y -> (b u v) c x y')
        x1 = rearrange(x1, 'b c u v x y -> (b u v) c x y')
        x2 = rearrange(x2, 'b c u v x y -> (b u v) c x y')
        xd1 = self.up_1(xd1)
        xd2 = self.up_2(xd2)
        x1 = self.down1(self.ca1(torch.cat((x1,xd1),dim = 1)))
        del xd1
        x2 = self.down2(self.ca2(torch.cat((x2,xd2),dim = 1)))
        del xd2
        #del x_g, x0, x1, x2

        return rearrange(x1, '(b u v) c x y -> b c u v x y', b=batch, c=channel, u=height_view-2, v=width_view-2),\
                rearrange(x2, '(b u v) c x y -> b c u v x y', b=batch, c=channel, u=height_view-2, v=width_view-2)



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask < 0.5, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn



class attention_part(nn.Module):
    def __init__(self, channel=32, scale=4):
        super(attention_part, self).__init__()

        layers = list()
        if channel == 1:
            for _ in range(4):
                layers.append(nn.Conv2d(1, 1, kernel_size=3, padding=1))
                #layers.append(nn.Conv3d(1, 1, kernel_size=(3, 3, 1), padding=(1, 1, 0)))
                layers.append(nn.ReLU())
        while channel != 1:
            if channel // scale == 0:
                layers.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1))
                #layers.append(nn.Conv3d(channel, 1, kernel_size=(3, 3, 1), padding=(1, 1, 0)))
                layers.append(nn.ReLU())
                break
            layers.append(nn.Conv2d(channel, channel // scale, kernel_size=3, padding=1))
            #layers.append(nn.Conv3d(channel, channel // scale, kernel_size=(3, 3, 1), padding=(1, 1, 0)))
            layers.append(nn.ReLU())
            channel //= scale
        self.seqn = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        x = rearrange(x, 'b c u v x y -> (b u v) c x y')
        #x = x.reshape(batch, channel, height_view, width_view, height * width)
        weight = self.seqn(x)
        weight = rearrange(weight, '(b u v) c x y -> b c (u v) x y', b=batch, u=height_view, v=width_view)
        weight = weight.view(batch, 1, height_view * width_view, height, width)
        weight = self.softmax(weight)
        x = rearrange(x, '(b u v) c x y -> b c (u v) x y', b=batch, c=channel, u=height_view, v=width_view)
        #x = x.view(batch, channel, height_view * width_view, height, width)
        x = torch.mul(x, weight)
        x = torch.sum(x, dim=2)

        return x



class OcclusionAwareAttentionBlock1(nn.Module):
    def __init__(self, channel=32, scale=16):
        super(OcclusionAwareAttentionBlock1, self).__init__()

        #self.OANet = OcclusionAttentionNetwork(n_head=2, d_model=channel, d_k=channel, d_v=channel)

        self.ColorAggregationNet = nn.Sequential(
            attention_part(channel, scale=scale),
            nn.ReLU(),
            # nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(channel, 3, kernel_size=3, padding=1, stride=1),
            # nn.ReLU()
        )

        self.AlphaAggregationNet = nn.Sequential(
            attention_part(channel, scale=scale),
            nn.ReLU(),
            # nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1),
            # # nn.ReLU()
            # nn.Sigmoid()
        )
    def forward(self, x):
        b, c, u, v, h, w = list(x.shape)
        mpi = self.ColorAggregationNet(x)
        mask = self.AlphaAggregationNet(x)
        return mpi, mask


class get_mpi(nn.Module):
    def __init__(self, channel=32):
        super(get_mpi, self).__init__()
        self.get_color = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(channel//2, 3, kernel_size=3, padding=1, stride=1),
            nn.ReLU())
        self.get_alpha = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(channel//2, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        return self.get_color(x1), self.get_alpha(x2)


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

class N2F_MPI(nn.Module):
    def __init__(self, channel=3):
        super(N2F_MPI, self).__init__()
        self.OAAB = OcclusionAwareAttentionBlock1(channel=channel, scale=4)
        self.OAAB1 = OcclusionAwareAttentionBlock1(channel=channel, scale=4)
        self.getmpib1 = get_mpi(channel)
        self.getmpib2 = get_mpi(channel)
        self.getmpif1 = get_mpi(channel)
        self.getmpif2 = get_mpi(channel)
        # self.layerffr = LayerFFR(channel)
        self.mugi1 = MuGIBlocks1(channel)
        self.mugi2 = MuGIBlocks1(channel)
        self.mugi3 = MuGIBlocks1(channel)
        self.mugi4 = MuGIBlocks1(channel)
        self.df1 = Dconv_Feature(channel,True)
        self.df2 = Dconv_Feature(channel,True)
        self.df3 = Dconv_Feature(channel,True)
        self.df4 = Dconv_Feature(channel,True)
        self.HWAB_t1 = HWABS1(n_feat=channel,o_feat=channel)
        self.HWAB_t2 = HWABS1(n_feat=channel,o_feat=channel)
        self.HWAB_f1 = HWABS1(n_feat=channel,o_feat=channel)
        self.HWAB_f2 = HWABS1(n_feat=channel,o_feat=channel)
    def forward(self, x1, x2,  disparity_list):
        batch, level,  channel, height_view, width_view, height, width = list(x1.shape)
        x1 = rearrange(x1, 'b l c u v x y -> (b l) c u v x y')
        x2 = rearrange(x2, 'b l c u v x y -> (b l) c u v x y')
        mpi_t, mask_t= self.OAAB(x1)
        mpi_f,mask_f = self.OAAB1(x2)
        mpi_t1,mask_t1 = self.getmpib1(mpi_t, mask_t)
        mpi_f1,mask_f1 = self.getmpif1(mpi_f, mask_f)
        # mask_t = rearrange(mask_t, '(b l) c x y -> b l c x y', b=batch)
        # mask_f = rearrange(mask_f, '(b l) c x y -> b l c x y', b=batch)
        # mpi_t2,mask_t2, mpi_f2, mask_f2 = self.layerffr(mpi_t, mask_t, mpi_f, mask_f)
        mpi_t2, mpi_f2 = self.mugi1(mpi_t, mpi_f)
        mask_t2, mask_f2 = self.mugi2(mask_t, mask_f)


        mpi_t2, mpi_f2 = self.HWAB_t1(mpi_t2,mpi_f2)
        mask_t2, mask_f2 = self.HWAB_t2(mask_t2,mask_f2)

        
        mpi_t2 = self.df1(rearrange(mpi_t2,'(b l) c h w -> b l c h w',b=batch,l=level))
        mask_t2 = self.df2(rearrange(mask_t2,'(b l) c h w -> b l c h w',b=batch,l=level))
        mpi_f2 = self.df3(rearrange(mpi_f2,'(b l) c h w -> b l c h w',b=batch,l=level))
        mpi_f2 = self.df3(rearrange(mpi_f2,'(b l) c h w -> b l c h w',b=batch,l=level))
        mask_f2 = self.df4(rearrange(mask_f2,'(b l) c h w -> b l c h w',b=batch,l=level))


        mpi_t2, mpi_f2 = self.HWAB_f1(mpi_t2,mpi_f2)
        mask_t2, mask_f2 = self.HWAB_f2(mask_t2,mask_f2)  

        mpi_t2, mpi_f2 = self.mugi3(mpi_t2, mpi_f2)
        mask_t2, mask_f2 = self.mugi4(mask_t2, mask_f2)


        mpi_t2,mask_t2 = self.getmpib1(mpi_t2, mask_t2)
        mpi_f2,mask_f2 = self.getmpif1(mpi_f2, mask_f2)
        del mpi_t, mask_t, mpi_f, mask_f
        mpi_t1 = rearrange(mpi_t1, '(b l) c x y -> b l c x y', b=batch)
        mask_t1 = rearrange(mask_t1, '(b l) c x y -> b l c x y', b=batch)
        mpi_f1 = rearrange(mpi_f1, '(b l) c x y -> b l c x y', b=batch)
        mask_f1 = rearrange(mask_f1, '(b l) c x y -> b l c x y', b=batch)
        mpi_t2 = rearrange(mpi_t2, '(b l) c x y -> b l c x y', b=batch)
        mask_t2 = rearrange(mask_t2, '(b l) c x y -> b l c x y', b=batch)
        mpi_f2 = rearrange(mpi_f2, '(b l) c x y -> b l c x y', b=batch)
        mask_f2 = rearrange(mask_f2, '(b l) c x y -> b l c x y', b=batch)
        return mpi_t1, mask_t1, mpi_f1, mask_f1, mpi_t2, mask_t2, mpi_f2, mask_f2
        # mpi_t = rearrange(mpi_t, '(b l) c x y -> b l c x y', b=batch)
        # mask_t = rearrange(mask_t, '(b l) c x y -> b l c x y', b=batch)
        # mpi_f = rearrange(mpi_f, '(b l) c x y -> b l c x y', b=batch)
        # mask_f = rearrange(mask_f, '(b l) c x y -> b l c x y', b=batch)

        # return mpi_t, mask_t, mpi_f, mask_f



class res_block_2d(nn.Module):
    def __init__(self, channel):
        super(res_block_2d, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        x = x + self.PRelu(self.conv(x))
        return x


class ResNet(nn.Module):
    def __init__(self, block_num, channel, in_channel=3,out_channel=3):
        super(ResNet, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_channel, channel, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.PReLU())
        for _ in range(block_num):
            layers.append(res_block_2d(channel))
        self.res_conv = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.res_conv(x)
        x = self.final_conv(x)
        return x


class RenderNet(nn.Module):
    def __init__(self, block_num, channel,if_train=False, in_channel=3,out_channel=3):
        super(RenderNet, self).__init__()
        if if_train:
            self.ecnn = ECNN()
            self.icnn = ICNN()
        self.if_train = if_train
    def forward(self, color, alpha, target_position_list, view_n_new, disparity_list):
        target_view_all = []
        if self.if_train:
            alpha = rearrange(alpha, 'b l c x y -> (b l) c x y')
            color = rearrange(color, 'b l c x y -> (b l) c x y')
            # alpha_ = self.ecnn(torch.cat((color, alpha), 1))
            # alpha = alpha + alpha_
            # color_ = self.icnn(torch.cat((color, alpha), 1))
            # color = color + color_
            alpha = self.ecnn(torch.cat((color, alpha), 1))
            color = self.icnn(torch.cat((color, alpha), 1))
            alpha = rearrange(alpha, '(b l) c x y -> b l c x y', l = len(disparity_list))
            color = rearrange(color, '(b l) c x y -> b l c x y', l = len(disparity_list))

        for target_position in target_position_list:

            target_view, _ = self.get_view_fast(color, alpha, target_position, view_n_new, disparity_list)

            target_view_all.append(target_view.permute(0, 2, 3, 1))
        _, alpha_final = self.get_view_fast(color, alpha, [view_n_new//2,view_n_new//2], view_n_new, disparity_list)
        #target_view_all = torch.cat(target_view_all, 0)

        #target_view_all_final = self.ResNet(target_view_all)

        #target_view_all = target_view_all.permute(0, 2, 3, 1)
        #target_view_all_final = target_view_all_final.permute(0, 2, 3, 1)

        return target_view_all, alpha_final


    def get_view_fast(self, color, alpha, target_position, view_n_new, disparity_list):
        B, l, c, X, Y = color.shape
        disparity_grad = disparity_list[1] - disparity_list[0]
        central_position = [view_n_new // 2, view_n_new // 2]

        if central_position[0] != target_position[0] or central_position[1] != target_position[1]:
            mpi_and_mask = self.warp_all(torch.cat((color, alpha), 2), disparity_list, disparity_grad, central_position,
                                         target_position)

            mpi_and_mask = rearrange(mpi_and_mask, 'l (b c) x y -> b l c x y', b=B)

            color = mpi_and_mask[:, :, :-1]
            alpha = mpi_and_mask[:, :, -1:]

        view = torch.zeros(B, c, X, Y).cuda()
        mask_tmp = torch.ones(B, 1, X, Y).cuda()
        mask_sum = torch.zeros(B, 1, X, Y).cuda()

        for i in range(l):

            # if disparity_list[i] >= occ_disparity_plane:
            #     continue

            tmp = alpha[:, i] * mask_tmp
            view = view + color[:, i] * tmp

            mask_tmp = mask_tmp - tmp
            mask_sum = mask_sum + tmp

        view = view / (mask_sum + 1e-9)
        alpha_final = alpha/(torch.unsqueeze(mask_sum, dim=1) + 1e-9)

        return view, alpha_final

    def warp_all(self, img, disparity_list, disparity_grad, ori_position, novel_position):
        img = img.permute(1, 0, 2, 3, 4)
        img = img.reshape(disparity_list.shape[0], -1, img.shape[3], img.shape[4])
        ori_position = np.array(ori_position)
        novel_position = np.array(novel_position)
        theta = []
        for i, disparity in enumerate(disparity_list):
            d = (novel_position - ori_position) * (disparity + disparity_grad) * 2
            theta_t = torch.FloatTensor([[1, 0, d[1] / img.shape[3]], [0, 1, d[0] / img.shape[2]]]).cuda()
            theta.append(theta_t.unsqueeze(0))
        theta = torch.cat(theta, 0)
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        img = F.grid_sample(img, grid, mode='bilinear', align_corners=False)
        #img = F.grid_sample(img, grid, mode='nearest', align_corners=False)
        return img


class RRMPI(nn.Module):
    def __init__(self, channel=32, block_num=4):
        super(RRMPI, self).__init__()
        self.pre_conv = pre_part(channel)
        self.N2F_MPI = N2F_MPI(channel)
        self.SAS = MultiSAS(channel)
        self.RenderModule = RenderNet(block_num, channel)
        self.RenderModuleb = RenderNet(block_num, channel,if_train=True)
        self.RenderModulef = RenderNet(block_num, channel,if_train=True)
        print(channel)

    def forward(self, x,target_position_list, view_n_new, disparity_list):
        batch, level, height_view, width_view, height, width, c = list(x.shape)
        x = torch.flip(x,dims=[1])
        disparity_list = disparity_list[::-1]
        x = rearrange(x, 'b l u v h w c -> b l u v c h w')
        x = self.pre_conv(x)
        # b*l c u v h w
        #x = self.SAS(x)
        x1, x2 = self.SAS(x)
        del x
        x1 = rearrange(x1, '(b l) c u v h w -> b l c u v h w', b=batch)
        x2 = rearrange(x2, '(b l) c u v h w -> b l c u v h w', b=batch)
        #batch, level, channel, height_view, width_view, height, width = list(x.shape)
        color_bs, alpha_bs, color_fs, alpha_fs,color_b, alpha_b, color_f, alpha_f = self.N2F_MPI(x1,x2, disparity_list)
        del x1, x2
        
        target_view_list_bs, alpha_bs_final = self.RenderModule(color_b, alpha_b, target_position_list, view_n_new,
                                                                   disparity_list)
        target_view_list_fs, alpha_fs_final = self.RenderModule(color_f, alpha_f, target_position_list, view_n_new,
                                                                     disparity_list)
        target_view_list_bs2, alpha_bs2_final = self.RenderModule(color_bs, alpha_bs, target_position_list, view_n_new,
                                                                   disparity_list)
        target_view_list_fs2, alpha_fs2_final = self.RenderModule(color_fs, alpha_fs, target_position_list, view_n_new,
                                                                     disparity_list)
        target_view_list_b, alpha_b_final = self.RenderModuleb(color_b, alpha_b, target_position_list, view_n_new,
                                                                   disparity_list)#, occlusion_disparity_plane
        target_view_list_f, alpha_f_final = self.RenderModulef(color_f, alpha_f, target_position_list, view_n_new,
                                                                   disparity_list)

        return target_view_list_b, target_view_list_f,color_b, alpha_bs_final, color_f, alpha_fs_final,target_view_list_bs, target_view_list_fs,target_view_list_bs2, target_view_list_fs2



