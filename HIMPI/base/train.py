
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import torch
from torch.utils.data import DataLoader
from evaluation_index.evaluation_index import ProgressMeter, Meter, StrMeter, AverageMeterJustValue, \
    AverageMeterJustAVG, visdomMeter
import time  
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from losses.vgg import Vgg19
from initializers.initializers import weights_init
from scheduler.myscheduler import myLR, clip_gradient
from utils import utils
from math import log10
from train_options import TrainOptions
from losses.myloss import MY_LOSS_BR2
from dataload.dataload import DeRefLF_Train_Dataset, DeRefLF_Test_Dataset_New
import random
from utils.utils import result_save_all1
from model.HIMPI import RRMPI


def train_main():
    # Train parameters
    opt = TrainOptions().parse()
    device_ids = opt.gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    print(device_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ''' Define Model(set parameters)'''
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    start_time = time.strftime('%m%d%H', time.localtime(time.time()))

    model_name = opt.model_name
    model = globals()[model_name](opt.channel)

    utils.get_parameter_number(model)
    model.apply(weights_init('xavier'))

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.base_lr)
    scheduler = myLR(optimizer)

    ''' Create the save path for training models and record the training and test loss'''
    if opt.task_name is not None:
        ''' Loading the trained model'''
        path_checkpoint = f'./NetworkSave/{opt.tag}' + f'/{opt.task_name}/{opt.model_name}_{opt.current_iter}.pkl'
        print('Loading the trained model', path_checkpoint) 
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opt.current_iter = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        dir_model = f'./NetworkSave/{opt.tag}/{opt.task_name}'
    else:
        task_name = f'{opt.model_name}_{opt.tag}_{opt.crop_size}_{opt.up_size}_{start_time}'
        dir_model = f'./NetworkSave/{opt.tag}/{task_name}'

        print('dir_model:', dir_model)
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        else:
            print('Exist!')

    best_psnr = 0
    test_psnr = 0
    best_psnr_new = 0
    LOSS = visdomMeter('loss', 1)
    PSNR = visdomMeter('PSNR', 10)

    # load training dataset
    train_dataload_name = opt.train_dataload_name
    train_dataset = globals()[train_dataload_name](opt)
    # train_dataset =[]

    # load test dataset
    test_dataload_name = opt.test_dataload_name
    test_dataset = globals()[test_dataload_name](opt,device)
    #test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    #test_dataset = [[np.ones(7,7,280,448,3)], [np.ones(280,448,3)], [np.ones(280,448,3)], ["0"]]
    vgg = Vgg19(requires_grad=False).to(device)
    disparity_list = np.arange(opt.min, opt.max + opt.step, opt.step)
    for epoch in range(opt.current_iter, opt.MAX_EPOCH):

        ''' Validation during the training process'''
        if epoch % 10 == 0:
            # best_psnr_new, test_psnr = 0,0
            best_psnr_new, test_psnr = test_res(epoch, test_dataset, model, best_psnr, device, opt,disparity_list,dir_model)
            if len(opt.gpu_ids) > 1:
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }
            else:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }
            torch.save(checkpoint, dir_model + f'/{opt.model_name}_{epoch}.pkl')

            if best_psnr_new > best_psnr:
                torch.save(checkpoint, dir_model + f'/{opt.model_name}_best.pkl')

                best_psnr = best_psnr_new
            torch.cuda.empty_cache()
        ''' Training begin'''
        train_loss = train_res(train_dataset, model, epoch, optimizer, device, opt, vgg,disparity_list)
        LOSS.update(train_loss)
        PSNR.update(test_psnr)
        scheduler.step()


def train_res(train_dataset, model, epoch, optimizer, device, opt, vgg,disparity_list):
    time_start = time.time()
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    mid_time = time.time()
    angRes = opt.height_view
    target_position_list = []
    for i in range(0,opt.views):
        target_position = [random.randint(0, angRes-1), random.randint(0, angRes-1)]
        target_position_list.append(target_position)
    epoch_time = Meter('Time', ':6.3f')
    LOSS = AverageMeterJustAVG('Loss', ':3.6f')
    B_PSNR = AverageMeterJustAVG('B_PSNR', ':6.3f')
    R_PSNR = AverageMeterJustAVG('R_PSNR', ':6.3f')
    LR = Meter('LR', ':.2e')

    B_L1_LOSS = AverageMeterJustAVG('B_L1_loss', ':3.6f')
    B_Gradient_LOSS = AverageMeterJustAVG('B_Gradient_loss', ':3.6f')
    B_Vgg_LOSS = AverageMeterJustAVG('B_Vgg_loss', ':3.6f')
    B1_L1_LOSS = AverageMeterJustAVG('B1_L1_loss', ':3.6f')
    B2_L1_LOSS = AverageMeterJustAVG('B2_L1_loss', ':3.6f')
    R_L1_LOSS = AverageMeterJustAVG('R_L1_loss', ':3.6f')
    R_Gradient_LOSS = AverageMeterJustAVG('R_Gradient_loss', ':3.6f')
    R_Vgg_LOSS = AverageMeterJustAVG('R_Vgg_loss', ':3.6f')
    R1_L1_LOSS = AverageMeterJustAVG('R1_L1_loss', ':3.6f')
    R2_L1_LOSS = AverageMeterJustAVG('R2_L1_loss', ':3.6f')
    I_L1_LOSS = AverageMeterJustAVG('I_L1_loss', ':3.6f')
    Exclusion_LOSS = AverageMeterJustAVG('Exclusion_loss', ':3.6f')

    progress = ProgressMeter(opt.MAX_EPOCH,
                             [LR, LOSS, B_PSNR, R_PSNR, epoch_time, B_L1_LOSS, B_Gradient_LOSS, B_Vgg_LOSS, B1_L1_LOSS,B2_L1_LOSS, R_L1_LOSS,
                              R_Gradient_LOSS, R_Vgg_LOSS, R1_L1_LOSS, R2_L1_LOSS, I_L1_LOSS, Exclusion_LOSS],
                             prefix=f'Epoch: [{epoch}]')

    optimizer.zero_grad()
    for i, (train_data, B_gt_data, R_gt_data, train_psv, factor) in enumerate(train_loader):
        train_psv, factor =  train_psv.to(device), factor.to(device)
        gt_data,B_gt_data,R_gt_data = utils.get_target_view(train_data,B_gt_data,R_gt_data,target_position_list)
        # Forward pass: Compute predicted y by passing x to the model
        
        

        B_pred_data, R_pred_data, color_B, alpha_B,color_F, alpha_F, B_pred_step1, R_pred_step1,B_pred_step2, R_pred_step2 = model(train_psv,  target_position_list,
                                           opt.height_view, disparity_list)

        loss, B_L1_loss, B_Gradient_loss, B_Vgg_loss,B1_L1_loss,B2_L1_loss, R_L1_loss, R_Gradient_loss, R_Vgg_loss, R1_L1_loss,R2_L1_loss, I_L1_loss, Exclusion_loss = MY_LOSS_BR2(
            B_pred_data, B_gt_data, R_pred_data, R_gt_data, B_pred_step1, R_pred_step1, B_pred_step2, R_pred_step2, factor, device, opt, vgg)



        B_loss_mse = torch.nn.MSELoss().to(device)
        R_loss_mse = torch.nn.MSELoss().to(device)
        B_loss_mse2,R_loss_mse2 =  torch.zeros(1).cuda(), torch.zeros(1).cuda()
        for k in range(0,opt.views):
            B_loss_mse2 += B_loss_mse(B_pred_data[k], B_gt_data[k])
            R_loss_mse2 += R_loss_mse(R_pred_data[k], R_gt_data[k])
        B_psnr = 10 * log10(opt.views / B_loss_mse2.item())
        B_psnr = torch.from_numpy(np.array(B_psnr))
        R_psnr = 10 * log10(opt.views / R_loss_mse2.item())
        R_psnr = torch.from_numpy(np.array(R_psnr))

        LOSS.update(loss.item(), train_data.size(0))

        B_PSNR.update(B_psnr.item(), train_data.size(0))
        R_PSNR.update(R_psnr.item(), train_data.size(0))

        B_L1_LOSS.update(B_L1_loss.item(), train_data.size(0))
        R_L1_LOSS.update(R_L1_loss.item(), train_data.size(0))
        I_L1_LOSS.update(I_L1_loss.item(), train_data.size(0))

        B1_L1_LOSS.update(B1_L1_loss.item(), train_data.size(0))
        R1_L1_LOSS.update(R1_L1_loss.item(), train_data.size(0))

        B2_L1_LOSS.update(B2_L1_loss.item(), train_data.size(0))
        R2_L1_LOSS.update(R2_L1_loss.item(), train_data.size(0))

        B_Gradient_LOSS.update(B_Gradient_loss.item(), train_data.size(0))
        R_Gradient_LOSS.update(R_Gradient_loss.item(), train_data.size(0))
        
        Exclusion_LOSS.update(Exclusion_loss.item(), train_data.size(0))

        B_Vgg_LOSS.update(B_Vgg_loss.item(), train_data.size(0))
        R_Vgg_LOSS.update(R_Vgg_loss.item(), train_data.size(0))

        loss = loss / opt.accumulation_steps
        loss.backward()

        if (i + 1) % opt.accumulation_steps == 0:
            if opt.clip_value is not None:
                clip_gradient(optimizer, opt.clip_value)
            optimizer.step()
            optimizer.zero_grad()

    LR.update(optimizer.state_dict()['param_groups'][0]['lr'])
    epoch_time.update(time.time() - time_start)

    progress.display(epoch)
    return LOSS.avg


def test_res(epoch,test_dataset, model, best_psnr, device, opt,disparity_list,dir_model):
    time_start = time.time()
    model.eval()
    border_crop = 15
    [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list,psv_list] = test_dataset
    num = len(image_path_list)
    B_PSNR = AverageMeterJustValue('B_PSNR', ':6.4f')
    B_SSIM = AverageMeterJustValue('B_SSIM', ':6.4f')
    R_PSNR = AverageMeterJustValue('R_PSNR', ':6.4f')
    R_SSIM = AverageMeterJustValue('R_SSIM', ':6.4f')
    B1_PSNR = AverageMeterJustValue('B1_PSNR', ':6.4f')
    R1_PSNR = AverageMeterJustValue('R1_PSNR', ':6.4f')
    IMG_NAME = StrMeter('IMG_NAME', ':s')
    progress = ProgressMeter(num, [B_PSNR, B_SSIM, R_PSNR, R_SSIM, B1_PSNR, R1_PSNR, IMG_NAME], prefix='Test: ')
    for idx in range(num):
        train_data = train_data_list[idx]
        B_gt_data = B_gt_data_list[idx]
        R_gt_data = R_gt_data_list[idx]
        image_path = image_path_list[idx]
        psv = psv_list[idx].to(device)
        target_position_list = []
        # for i in range(0,opt.height_view):
        #     for j in range(0,opt.height_view):
        #         target_position_list.append([i,j])
        for i in range(0,5):
            target_position_list.append([random.randint(0, opt.height_view-1), random.randint(0, opt.height_view-1)])
        gt_data,B_gt_data,R_gt_data = utils.get_target_view_n(train_data,B_gt_data,R_gt_data,target_position_list)
        B_psnr =0
        B_ssim =0
        R_psnr =0
        R_ssim =0
        B1_psnr =0
        R1_psnr =0
        with torch.no_grad():
            # Forward pass: Compute predicted y by passing x to the model
            u, v, h, w, c = train_data.shape
            h_res = h % 8
            w_res = w % 8
            h = h - h_res
            w = w - w_res
            border_crop_h = border_crop - h_res
            border_crop_w = border_crop - w_res
            B_gt_pred, R_gt_pred, color_B, alpha_B,color_F, alpha_F, B1_gt_pred, R1_gt_pred,B2_gt_pred, R2_gt_pred  = \
                utils.test_crop1(psv[:, :, :,:, 0:h, 0:w, :], model, opt.height_view, target_position_list, disparity_list, max_length=250, shave=8, mod=4)
            # l = []
            # l.append(torch.zeros(1,280,448,3).to(device))
            # B_gt_pred =l.copy()
            # R_gt_pred = l.copy()
            # color_B = torch.zeros(1,11,3,307,471).to(device)
            # alpha_B = torch.zeros(1,11,1,307,471).to(device)
            # color_F = torch.zeros(1,11,3,307,471).to(device)
            # alpha_F = torch.zeros(1,11,1,307,471).to(device)
            # B1_gt_pred = l.copy()
            # R1_gt_pred = l.copy()
            # #model(psv[:, :, :, 0:h, 0:w, :], target_position_list,opt.height_view, disparity_list)
            # B_gt_pred, R_gt_pred, _, _ = model(psv[:, :, :, 0:h, 0:w, :], target_position_list,opt.height_view, disparity_list)
            for k in range(len(target_position_list)):
                #print("resb",B_gt_pred[k].shape)
                B_gt_pred1 = B_gt_pred[k]
                # print("resb",B_gt_pred1.shape)
                # print("b",B_gt_data[k].shape)
                R_gt_pred1 = R_gt_pred[k]
                B1_gt_pred1 = B1_gt_pred[k]
                R1_gt_pred1 = R1_gt_pred[k]
                B_gt_pred_tmp = B_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
                B_gt_data1 = B_gt_data[k][border_crop:-border_crop, border_crop:-border_crop, :]
                B_gt_pred_tmp = np.clip(B_gt_pred_tmp, 0, 1)
                B_gt_data1 = np.clip(B_gt_data1, 0, 1)

                R_gt_pred_tmp = R_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
                R_gt_data1 = R_gt_data[k][border_crop:-border_crop, border_crop:-border_crop, :]
                R_gt_pred_tmp = np.clip(R_gt_pred_tmp, 0, 1)
                R_gt_data1 = np.clip(R_gt_data1, 0, 1)

                B1_gt_pred_tmp = B1_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
                B1_gt_pred_tmp = np.clip(B1_gt_pred_tmp, 0, 1)

                R1_gt_pred_tmp = R1_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
                R1_gt_pred_tmp = np.clip(R1_gt_pred_tmp, 0, 1)

                #print(type(B_gt_pred_tmp),type(B_gt_data1))
                B_psnr += peak_signal_noise_ratio(B_gt_pred_tmp, B_gt_data1, data_range=1)
                B_ssim += structural_similarity(B_gt_pred_tmp, B_gt_data1, data_range=1, multichannel=True)
                R_psnr += peak_signal_noise_ratio(R_gt_pred_tmp, R_gt_data1, data_range=1)
                R_ssim += structural_similarity(R_gt_pred_tmp, R_gt_data1, data_range=1, multichannel=True)
                B1_psnr += peak_signal_noise_ratio(B1_gt_pred_tmp, B_gt_data1, data_range=1)
                R1_psnr += peak_signal_noise_ratio(R1_gt_pred_tmp, R_gt_data1, data_range=1)
            save_dir = dir_model+opt.dir_save+'_epoch_'+str(epoch)+'/'
            result_save_all1(opt, save_dir, image_path, B_gt_pred, R_gt_pred,B1_gt_pred,R1_gt_pred,B_gt_data,R_gt_data,color_B,alpha_B,color_F,alpha_F,target_position_list)
            B_PSNR.update(B_psnr/len(target_position_list))
            B_SSIM.update(B_ssim/len(target_position_list))
            R_PSNR.update(R_psnr/len(target_position_list))
            R_SSIM.update(R_ssim/len(target_position_list))
            B1_PSNR.update(B1_psnr/len(target_position_list))
            R1_PSNR.update(R1_psnr/len(target_position_list))
            IMG_NAME.update(image_path)
        progress.display(idx)
    PSNR = (B_PSNR.avg + R_PSNR.avg) / 2
    if PSNR > best_psnr:
        best_psnr = PSNR

    print(
        f'===> PSNR:{PSNR:.4f} dB / BEST {best_psnr:.4f} dB '
        f'B_PSNR: {B_PSNR.avg:.4f} dB B_SSIM: {B_SSIM.avg:.4f} R_PSNR: {R_PSNR.avg:.4f} dB R_SSIM: {R_SSIM.avg:.4f} '
        f'Time: {time.time() - time_start:.6f}')
    return best_psnr, PSNR


if __name__ == '__main__':
    train_main()
