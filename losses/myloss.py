import torch
from torch import nn
import torch.nn.functional as F
from losses.vgg import VGGLoss

from math import exp

def L1_loss(pred_data, gt_data, device, opt, loss_count=0):
    criterion = torch.nn.L1Loss().to(device)
    l1_loss = opt.loss_weight[loss_count] * criterion(pred_data, gt_data)
    return l1_loss


def Gradient_loss(pred_data, gt_data, device, opt, loss_count=0):
    grad_loss = opt.loss_weight[loss_count] * gradient_loss(pred_data, gt_data, device)
    return grad_loss


def Vgg_loss(vgg, pred_data, gt_data, device, opt, loss_count=0):
    loss_vgg = VGGLoss(device=device, vgg=vgg, normalize=False)
    pred_data_tmp = pred_data.permute(0, 3, 1, 2).contiguous()
    gt_data_tmp = gt_data.permute(0, 3, 1, 2).contiguous()
    vgg_loss = opt.loss_weight[loss_count] * loss_vgg(pred_data_tmp, gt_data_tmp)

    return vgg_loss

def Exclusion_loss(pred_b,pred_f,device,opt,loss_count =0):
    loss_exc = ExclusionLoss().to(device=device)
    exc_loss = opt.loss_weight[loss_count]*loss_exc(pred_b.permute(0,3,1,2),pred_f.permute(0,3,1,2))
    return exc_loss


def AlphaExclusion_loss(alpha_b,alpha_f,device,opt,loss_count =0):
    loss_exc = AlphaExclusionLoss().to(device=device)
    exc_loss = []
    for i in range(opt.disp_num):
        exc_loss.append(loss_exc(alpha_b[:,i],alpha_f[:,i]))
    exc_loss = opt.loss_weight[loss_count]*sum(exc_loss)/opt.disp_num
    return exc_loss

class Gradient_Net(nn.Module):
    def __init__(self, device):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


def gradient(x, device):
    b, h, w, c = x.shape
    #  rgb2gray       rgb 0.3 , 0.59 , 0.11
    if c == 3:
        x = 0.3 * x[:, :, :, 0] + 0.59 * x[:, :, :, 1] + 0.11 * x[:, :, :, 2]
    x = x.unsqueeze(1)
    gradient_model = Gradient_Net(device).to(device)
    g = gradient_model(x)
    return g


def gradient_loss(pred_data, gt_data, device):
    pred_data_gradient = gradient(pred_data, device)
    gt_data_gradient = gradient(gt_data, device)
    criterion = torch.nn.L1Loss().to(device)
    loss = criterion(pred_data_gradient, gt_data_gradient)
    return loss


class ExclusionLoss(nn.Module):
    def __init__(self, level=3, eps=1e-6):
        super().__init__()
        self.level = level
        self.eps = eps

    def forward(self, img_T, img_R):
        grad_x_loss = []
        grad_y_loss = []

        for l in range(self.level):
            grad_x_T, grad_y_T = compute_grad(img_T)
            grad_x_R, grad_y_R = compute_grad(img_R)

            alphax = (2.0 * torch.mean(torch.abs(grad_x_T))) / (torch.mean(torch.abs(grad_x_R)) + self.eps)
            alphay = (2.0 * torch.mean(torch.abs(grad_y_T))) / (torch.mean(torch.abs(grad_y_R)) + self.eps)

            gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
            grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
            gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1

            grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + self.eps) ** 0.25)
            grad_y_loss.append(((torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2)))) + self.eps) ** 0.25)

            img_T = F.interpolate(img_T, scale_factor=0.5, mode='bilinear')
            img_R = F.interpolate(img_R, scale_factor=0.5, mode='bilinear')
        loss_gradxy = torch.sum(sum(grad_x_loss) / 3) + torch.sum(sum(grad_y_loss) / 3)

        return loss_gradxy / 2
    

def compute_grad(img):
    gradx = img[..., 1:, :,:] - img[..., :-1, :,:]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class AlphaExclusionLoss(nn.Module):
    def __init__(self, level=3, eps=1e-6):
        super().__init__()
        self.level = level
        self.eps = eps

    def forward(self, alpha_b, alpha_f):
        grad_x_loss = []

        for l in range(self.level):
            

            alphax = (2.0 * torch.mean(torch.abs(alpha_b))) / (torch.mean(torch.abs(alpha_f)) + self.eps)
            gradx1_s = (torch.sigmoid(alpha_b) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
            gradx2_s = (torch.sigmoid(alpha_f * alphax) * 2) - 1

            grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + self.eps) ** 0.25)

            alpha_b = F.interpolate(alpha_b, scale_factor=0.5, mode='bilinear')
            alpha_f = F.interpolate(alpha_f, scale_factor=0.5, mode='bilinear')
        loss_gradxy = torch.sum(sum(grad_x_loss) / 3)

        return loss_gradxy






def regularization_loss(model, weight_decay, p=2):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = torch.norm(param, p=p)
            reg_loss = reg_loss + l2_reg

    reg_loss = weight_decay * reg_loss
    return reg_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()



def MY_LOSS_BR2(B_pred_data, B_gt_data, R_pred_data, R_gt_data,B_pred_step1, R_pred_step1,B_pred_step2, R_pred_step2, factor, device, opt, vgg):
    factor = factor.reshape((len(factor), 1, 1, 1))
    count = len(B_pred_data)
    B_L1_loss, B_Gradient_loss, B_Vgg_loss, B1_L1_loss, B2_L1_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
    R_L1_loss, R_Gradient_loss, R_Vgg_loss, R1_L1_loss, R2_L1_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
    I_L1_loss, Exc_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
    for k in range(0,len(B_pred_data)):
        B_L1_loss += L1_loss(B_pred_data[k], B_gt_data[k], device, opt, loss_count=0)
        B_Gradient_loss += Gradient_loss(B_pred_data[k], B_gt_data[k], device, opt, loss_count=1)
        B_Vgg_loss += Vgg_loss(vgg, B_pred_data[k], B_gt_data[k], device, opt, loss_count=2)
        B1_L1_loss += L1_loss(B_pred_step1[k], B_gt_data[k], device, opt, loss_count=3)
        B2_L1_loss += L1_loss(B_pred_step2[k], B_gt_data[k], device, opt, loss_count=4)

        R_L1_loss += L1_loss(R_pred_data[k], R_gt_data[k], device, opt, loss_count=0)
        R_Gradient_loss += Gradient_loss(R_pred_data[k], R_gt_data[k], device, opt, loss_count=1)
        R_Vgg_loss += Vgg_loss(vgg, R_pred_data[k], R_gt_data[k], device, opt, loss_count=2)
        R1_L1_loss += L1_loss(R_pred_step1[k], R_gt_data[k], device, opt, loss_count=3)
        R2_L1_loss += L1_loss(R_pred_step2[k], R_gt_data[k], device, opt, loss_count=4)
        I_pred_data = B_pred_data[k] * factor + R_pred_data[k] * (1 - factor)
        I_gt_data = B_gt_data[k] * factor + R_gt_data[k] * (1 - factor)
        I_L1_loss += L1_loss(I_pred_data, I_gt_data, device, opt, loss_count=5)

        Exc_loss += Exclusion_loss(B_pred_data[k],R_pred_data[k], device, opt, loss_count=5)
        # Exc_loss += AlphaExclusion_loss(alpha_b,alpha_f, device, opt, loss_count=5)
    B_loss = B_L1_loss + B_Gradient_loss + B_Vgg_loss +B1_L1_loss+B2_L1_loss
    R_loss = R_L1_loss + R_Gradient_loss + R_Vgg_loss +R1_L1_loss+R2_L1_loss
    loss = B_loss + R_loss + I_L1_loss + Exc_loss

    return loss/count, B_L1_loss/count, B_Gradient_loss/count, B_Vgg_loss/count, B1_L1_loss/count,B2_L1_loss/count, R_L1_loss/count, R_Gradient_loss/count, R_Vgg_loss/count, R1_L1_loss/count, B2_L1_loss/count,I_L1_loss,Exc_loss/count