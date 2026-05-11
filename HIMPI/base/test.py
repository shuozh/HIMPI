import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import time
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from evaluation_index.evaluation_index import quality_assess, ProgressMeter, StrMeter, AverageMeterJustValue
from utils import utils
import torch.nn as nn
from test_options import TestOptions
from dataload.dataload import DeRefLF_Train_Dataset, DeRefLF_Test_Dataset_New
from utils.utils import result_save
from utils.utils import result_save_all1
from model.HIMPI import RRMPI

class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

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
        for i in range(0,opt.height_view):
            for j in range(0,opt.height_view):
                target_position_list.append([i,j])
        # for i in range(0,5):
        # target_position_list.append([3,3])
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
                utils.test_crop1(psv[:, :, :,:, 0:h, 0:w, :], model, opt.height_view, target_position_list, disparity_list, max_length=250, shave=32, mod=4)
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

# def test_res(epoch,test_dataset, model, best_psnr, device, opt,disparity_list,dir_model):
#     time_start = time.time()
#     model.eval()
#     border_crop = 15
#     [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list,psv_list] = test_dataset
#     num = len(image_path_list)
#     target_position_list = []
#     for i in range(0,7):
#         for j in range(0,7):
#             target_position_list.append([i,j])
#     #target_position_list =[[opt.height_view//2,opt.width_view//2]]
#     B_PSNR = AverageMeterJustValue('B_PSNR', ':6.4f')
#     B_SSIM = AverageMeterJustValue('B_SSIM', ':6.4f')
#     R_PSNR = AverageMeterJustValue('R_PSNR', ':6.4f')
#     R_SSIM = AverageMeterJustValue('R_SSIM', ':6.4f')
#     B1_PSNR = AverageMeterJustValue('B1_PSNR', ':6.4f')
#     R1_PSNR = AverageMeterJustValue('R1_PSNR', ':6.4f')
#     IMG_NAME = StrMeter('IMG_NAME', ':s')
#     progress = ProgressMeter(num, [B_PSNR, B_SSIM, R_PSNR, R_SSIM, B1_PSNR, R1_PSNR, IMG_NAME], prefix='Test: ')
#     for idx in range(num):
#         train_data = train_data_list[idx]
#         B_gt_data = B_gt_data_list[idx]
#         R_gt_data = R_gt_data_list[idx]
#         image_path = image_path_list[idx]
#         psv = psv_list[idx].to(device)
#         B_psnr =0
#         B_ssim =0
#         R_psnr =0
#         R_ssim =0
#         B1_psnr =0
#         R1_psnr =0
#         with torch.no_grad():
#             # Forward pass: Compute predicted y by passing x to the model
#             u, v, h, w, c = train_data.shape
#             h_res = h % 8
#             w_res = w % 8
#             h = h - h_res
#             w = w - w_res
#             border_crop_h = border_crop - h_res
#             border_crop_w = border_crop - w_res
#             # B_gt_pred, R_gt_pred, color_B, alpha_B,color_F, alpha_F, B1_gt_pred, R1_gt_pred  = \
#             #     model(psv[:, :, :,:, 0:h, 0:w, :], target_position_list=target_position_list, view_n_new=opt.height_view, disparity_list=disparity_list)
#             B_gt_pred, R_gt_pred, color_B, alpha_B,color_F, alpha_F, B1_gt_pred, R1_gt_pred,B2_gt_pred, R2_gt_pred  = \
#                 utils.test_crop1(psv[:, :, :,:, 0:h, 0:w, :], model, opt.height_view, target_position_list, disparity_list, max_length=250, shave=8, mod=4)
#                 #utils.test_crop(psv[:, :, :,:, 0:h, 0:w, :], model, opt.height_view, target_position_list, disparity_list, max_length=120, shave=8, mod=4)
#             for k in range(len(target_position_list)):
#                 #print("resb",B_gt_pred[k].shape)
#                 B_gt_pred1 = B_gt_pred[k]
#                 # print("resb",B_gt_pred1.shape)
#                 # print("b",B_gt_data[k].shape)
#                 R_gt_pred1 = R_gt_pred[k]
#                 B1_gt_pred1 = B1_gt_pred[k]
#                 R1_gt_pred1 = R1_gt_pred[k]
#                 B_gt_pred_tmp = B_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
#                 B_gt_data1 = B_gt_data[k][border_crop:-border_crop, border_crop:-border_crop, :]
#                 B_gt_pred_tmp = np.clip(B_gt_pred_tmp, 0, 1)
#                 B_gt_data1 = np.clip(B_gt_data1, 0, 1)

#                 R_gt_pred_tmp = R_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
#                 R_gt_data1 = R_gt_data[k][border_crop:-border_crop, border_crop:-border_crop, :]
#                 R_gt_pred_tmp = np.clip(R_gt_pred_tmp, 0, 1)
#                 R_gt_data1 = np.clip(R_gt_data1, 0, 1)

#                 B1_gt_pred_tmp = B1_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
#                 B1_gt_pred_tmp = np.clip(B1_gt_pred_tmp, 0, 1)

#                 R1_gt_pred_tmp = R1_gt_pred1[0, border_crop:-border_crop_h, border_crop:-border_crop_w,:].cpu().numpy()
#                 R1_gt_pred_tmp = np.clip(R1_gt_pred_tmp, 0, 1)

#                 #print(type(B_gt_pred_tmp),type(B_gt_data1))
#                 B_psnr += peak_signal_noise_ratio(B_gt_pred_tmp, B_gt_data1, data_range=1)
#                 B_ssim += structural_similarity(B_gt_pred_tmp, B_gt_data1, data_range=1, multichannel=True)
#                 R_psnr += peak_signal_noise_ratio(R_gt_pred_tmp, R_gt_data1, data_range=1)
#                 R_ssim += structural_similarity(R_gt_pred_tmp, R_gt_data1, data_range=1, multichannel=True)
#                 B1_psnr += peak_signal_noise_ratio(B1_gt_pred_tmp, B_gt_data1, data_range=1)
#                 R1_psnr += peak_signal_noise_ratio(R1_gt_pred_tmp, R_gt_data1, data_range=1)
#             save_dir = dir_model+opt.dir_save+'_epoch_'+str(epoch)+'/'
#             result_save_all1(opt, save_dir, image_path, B_gt_pred, R_gt_pred,B1_gt_pred,R1_gt_pred,B_gt_data,R_gt_data,color_B,alpha_B,color_F,alpha_F,target_position_list)
#             B_PSNR.update(B_psnr/len(target_position_list))
#             B_SSIM.update(B_ssim/len(target_position_list))
#             R_PSNR.update(R_psnr/len(target_position_list))
#             R_SSIM.update(R_ssim/len(target_position_list))
#             B1_PSNR.update(B1_psnr/len(target_position_list))
#             R1_PSNR.update(R1_psnr/len(target_position_list))
#             IMG_NAME.update(image_path)
#         progress.display(idx)
#     PSNR = (B_PSNR.avg + R_PSNR.avg) / 2
#     if PSNR > best_psnr:
#         best_psnr = PSNR

#     print(
#         f'===> PSNR:{PSNR:.4f} dB / BEST {best_psnr:.4f} dB '
#         f'B_PSNR: {B_PSNR.avg:.4f} dB B_SSIM: {B_SSIM.avg:.4f} R_PSNR: {R_PSNR.avg:.4f} dB R_SSIM: {R_SSIM.avg:.4f} '
#         f'Time: {time.time() - time_start:.6f}')
#     return best_psnr, PSNR

def test_main():
    opt = TestOptions().parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    device_ids = opt.gpu_ids
    if len(opt.gpu_ids) > 1:
        gpu_ids = opt.gpu_ids.split(',')
        for i in range(len(gpu_ids)):
            gpu_ids[i] = int(gpu_ids[i])
        device_ids = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print('=' * 40)
    print('create save directory...')
    expr_dir = os.path.join('./', opt.tag)
    if not os.path.exists(expr_dir):
        os.mkdir(expr_dir)
    dir_result = os.path.join(expr_dir, opt.dir_result)
    #sys.stdout = Logger(os.path.join(dir_result, f'test_{int(time.time())}.log'), sys.stdout)
    print('done')
    print('=' * 40)
    print('build network...')

    model_name = opt.model_name
    model = globals()[model_name](opt.channel)

    utils.get_parameter_number(model)
    if len(opt.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model.to(device)

    model.eval()
    print('done')
    print('=' * 40)
    print('load model...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1000, gamma=0.5)
    # dir_model = f'./{opt.tag}/{opt.dir_model}'
    # path_checkpoint = dir_model + f'{opt.model_name}_best.pkl'
    # checkpoint = torch.load(path_checkpoint)
    # # model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # opt.current_iter = checkpoint['epoch']
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    path_checkpoint = f'./NetworkSave/{opt.tag}' + f'/{opt.task_name}/{opt.model_name}_{opt.current_iter}.pkl'
    print('Loading the trained model', path_checkpoint) 
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    opt.current_iter = checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print('done')
    print('=' * 40)
    print('predict image...')
    test_dataset = DeRefLF_Test_Dataset_New(opt,device)
    # for index, image_name in enumerate(test_dataset):
    disparity_list = np.arange(opt.min, opt.max + opt.step, opt.step)
    dir_model = f'./NetworkSave/{opt.tag}/{opt.task_name}'
    test_res(0,test_dataset,model,0,device,opt,disparity_list,dir_model)
    print('all done')



if __name__ == '__main__':
    test_main()
