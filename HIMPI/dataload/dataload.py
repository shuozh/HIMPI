import numpy as np
import torch.utils.data as data
from utils import utils
import cv2
import torch
import os
class DeRefLF_Train_Dataset(data.Dataset):
    def __init__(self, opt):
        self.up_size = opt.up_size
        self.crop_size = opt.crop_size
        self.if_flip = opt.if_flip
        self.if_rotation = opt.if_rotation
        self.B_dir_train = opt.B_dir_train
        self.R_dir_train = opt.R_dir_train

        self.height_view = opt.height_view
        self.min = opt.min
        self.max = opt.max
        self.step = opt.step

        self.disparity = opt.disparity

        if self.R_dir_train == None:
            self.B_gt_data = utils.load_B_R_gt(self.B_dir_train, self.R_dir_train)
        else:
            self.B_gt_data, self.R_gt_data = utils.load_B_R_gt(self.B_dir_train, self.R_dir_train)

        self.num = len(self.B_gt_data)

    def __len__(self):
        return self.num * self.up_size

    def __getitem__(self, idx):
        B_idxtmp = idx % self.num
        while (1):
            if self.R_dir_train == None:
                index = np.random.randint(0, self.num)
                while index == B_idxtmp:
                    index = np.random.randint(0, self.num)
                flag = np.random.rand()
                if flag<0.5:
                    B_gt_data = self.B_gt_data[B_idxtmp]
                    R_gt_data = self.B_gt_data[index]
                else:
                    R_gt_data = self.B_gt_data[B_idxtmp]
                    B_gt_data= self.B_gt_data[index]
            else:
                index = np.random.randint(0, self.num)
                B_gt_data = self.B_gt_data[B_idxtmp]
                R_gt_data = self.R_gt_data[index]

            if np.mean(B_gt_data) * 1 / 2 > np.mean(R_gt_data):
                continue
            if np.mean(R_gt_data) * 1 / 2 > np.mean(B_gt_data):
                continue
            train_data, B_gt_data_c, R_gt_data_c, factor = utils.create_B_R(B_gt_data, R_gt_data,
                                                                            crop_size=self.crop_size + 2 * self.max * (
                                                                                    7 - 1),
                                                                            disparity=self.disparity)
            if B_gt_data_c.max() < 0.15 or R_gt_data_c.max() < 0.15 or train_data.max() < 0.1:
                continue
            break

        #focus_data = utils.create_F(train_data, self.min, self.max, self.step)
        focus_data = torch.from_numpy(train_data[np.newaxis, :, :, :, :, :])
        focus_data = utils.view_warp_different_fast(focus_data,self.min,self.max,self.step)
        focus_data = focus_data.squeeze(0).numpy()
        if self.height_view < 7:
            edge = int((7 - self.height_view) / 2)
            train_data = train_data[edge:edge + self.height_view, edge:edge + self.height_view, :]
            B_gt_data_c = B_gt_data_c[edge:edge + self.height_view, edge:edge + self.height_view, :]
            R_gt_data_c = R_gt_data_c[edge:edge + self.height_view, edge:edge + self.height_view, :]
            focus_data = focus_data[:,edge:edge + self.height_view, edge:edge + self.height_view, :]

        top = int(self.max * (self.height_view - 1))
        left = top
        focus_data, _, _ = utils.random_crop(focus_data, self.crop_size, self.crop_size, top, left)
        train_data, _, _ = utils.random_crop(train_data, self.crop_size, self.crop_size, top, left)
        B_gt_data_c, _, _ = utils.random_crop(B_gt_data_c, self.crop_size, self.crop_size, top, left)
        R_gt_data_c, _, _ = utils.random_crop(R_gt_data_c, self.crop_size, self.crop_size, top, left)
        if self.if_flip:
            train_data, random_tmp = utils.random_flip(train_data)
            B_gt_data_c, _ = utils.random_flip(B_gt_data_c, random_tmp)
            R_gt_data_c, _ = utils.random_flip(R_gt_data_c, random_tmp)
            focus_data, _ = utils.random_flip(focus_data, random_tmp)
        if self.if_rotation:
            train_data, random_tmp = utils.random_rotation(train_data)
            B_gt_data_c, _ = utils.random_rotation(B_gt_data_c, random_tmp)
            R_gt_data_c, _ = utils.random_rotation(R_gt_data_c, random_tmp)
            focus_data, _ = utils.random_rotation(focus_data, random_tmp)

        train_data = torch.from_numpy(train_data.copy())
        B_gt_data_c = torch.from_numpy(B_gt_data_c.copy())
        R_gt_data_c = torch.from_numpy(R_gt_data_c.copy())
        focus_data = torch.from_numpy(focus_data.copy())
        factor = torch.tensor(factor, dtype=torch.float32)

        return train_data, B_gt_data_c, R_gt_data_c,focus_data, factor

class DeRefLF_Train_Dataset_Cuda(data.Dataset):
    def __init__(self, opt):
        self.up_size = opt.up_size
        self.crop_size = opt.crop_size
        self.if_flip = opt.if_flip
        self.if_rotation = opt.if_rotation
        self.B_dir_train = opt.B_dir_train
        self.R_dir_train = opt.R_dir_train

        self.height_view = opt.height_view
        self.min = opt.min
        self.max = opt.max
        self.step = opt.step
        self.disparity = opt.disparity

        if self.R_dir_train == None:
            self.B_gt_data = utils.load_B_R_gt(self.B_dir_train, self.R_dir_train)
        else:
            self.B_gt_data, self.R_gt_data = utils.load_B_R_gt(self.B_dir_train, self.R_dir_train)

        self.num = len(self.B_gt_data)

    def __len__(self):
        return self.num * self.up_size

    def __getitem__(self, idx):
        B_idxtmp = idx % self.num
        while (1):
            if self.R_dir_train == None:
                index = np.random.randint(0, self.num)
                while index == B_idxtmp:
                    index = np.random.randint(0, self.num)
                flag = np.random.rand()
                if flag<0.5:
                    B_gt_data = self.B_gt_data[B_idxtmp]
                    R_gt_data = self.B_gt_data[index]
                else:
                    R_gt_data = self.B_gt_data[B_idxtmp]
                    B_gt_data= self.B_gt_data[index]
            else:
                index = np.random.randint(0, self.num)
                B_gt_data = self.B_gt_data[B_idxtmp]
                R_gt_data = self.R_gt_data[index]

            if np.mean(B_gt_data) * 1 / 2 > np.mean(R_gt_data):
                continue
            if np.mean(R_gt_data) * 1 / 2 > np.mean(B_gt_data):
                continue
            train_data, B_gt_data_c, R_gt_data_c, factor = utils.create_B_R(B_gt_data, R_gt_data,
                                                                            crop_size=self.crop_size + 2 * self.max * (
                                                                                    7 - 1),
                                                                            disparity=self.disparity)
            if B_gt_data_c.max() < 0.15 or R_gt_data_c.max() < 0.15 or train_data.max() < 0.1:
                continue
            break

        #focus_data = utils.create_F(train_data, self.min, self.max, self.step)
        # focus_data = torch.from_numpy(train_data[np.newaxis, :, :, :, :, :])
        # focus_data = utils.view_warp_different_fast(focus_data,self.min,self.max,self.step)
        # focus_data = focus_data.squeeze(0).numpy()
        focus_data = torch.from_numpy(train_data).cuda()
        focus_data = utils.view_warp_different_fast_cuda(focus_data,self.min,self.max,self.step)
        focus_data = focus_data.cpu().numpy()
        if self.height_view < 7:
            edge = int((7 - self.height_view) / 2)
            train_data = train_data[edge:edge + self.height_view, edge:edge + self.height_view, :]
            B_gt_data_c = B_gt_data_c[edge:edge + self.height_view, edge:edge + self.height_view, :]
            R_gt_data_c = R_gt_data_c[edge:edge + self.height_view, edge:edge + self.height_view, :]
            focus_data = focus_data[:,edge:edge + self.height_view, edge:edge + self.height_view, :]

        top = int(self.max * (self.height_view - 1))
        left = top
        focus_data, _, _ = utils.random_crop(focus_data, self.crop_size, self.crop_size, top, left)
        train_data, _, _ = utils.random_crop(train_data, self.crop_size, self.crop_size, top, left)
        B_gt_data_c, _, _ = utils.random_crop(B_gt_data_c, self.crop_size, self.crop_size, top, left)
        R_gt_data_c, _, _ = utils.random_crop(R_gt_data_c, self.crop_size, self.crop_size, top, left)
        if self.if_flip:
            train_data, random_tmp = utils.random_flip(train_data)
            B_gt_data_c, _ = utils.random_flip(B_gt_data_c, random_tmp)
            R_gt_data_c, _ = utils.random_flip(R_gt_data_c, random_tmp)
            focus_data, _ = utils.random_flip(focus_data, random_tmp)
        if self.if_rotation:
            train_data, random_tmp = utils.random_rotation(train_data)
            B_gt_data_c, _ = utils.random_rotation(B_gt_data_c, random_tmp)
            R_gt_data_c, _ = utils.random_rotation(R_gt_data_c, random_tmp)
            focus_data, _ = utils.random_rotation(focus_data, random_tmp)

        train_data = torch.from_numpy(train_data.copy())
        B_gt_data_c = torch.from_numpy(B_gt_data_c.copy())
        R_gt_data_c = torch.from_numpy(R_gt_data_c.copy())
        focus_data = torch.from_numpy(focus_data.copy())
        factor = torch.tensor(factor, dtype=torch.float32)

        return train_data, B_gt_data_c, R_gt_data_c,focus_data, factor

def DeRefLF_Test_Dataset_New(opt, device):
    [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list] = utils.load_test_new(opt.dir_test_images)
    psv_list = []
    if opt.height_view < 7:
        edge = int((7 - opt.height_view) / 2)
        train_data_list = [train_data[edge:edge + opt.height_view, edge:edge + opt.height_view, :] for train_data in train_data_list]
        B_gt_data_list = [B_gt_data[edge:edge + opt.height_view, edge:edge + opt.height_view, :] for B_gt_data in B_gt_data_list]
        R_gt_data_list = [R_gt_data[edge:edge + opt.height_view, edge:edge + opt.height_view, :] for R_gt_data in R_gt_data_list]
    for train_data in train_data_list:
        psv = train_data[np.newaxis, :, :, :, :, :]
        psv = torch.from_numpy(psv)
        psv = utils.view_warp_different_fast(psv, opt.min, opt.max, opt.step)
        psv_list.append(psv)
    return [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list, psv_list]




class DeRefLF_Test_Dataset_Cuda(data.Dataset):
    def __init__(self, opt):
        self.dir = opt.dir_test_images
        self.min = opt.min
        self.max = opt.max
        self.step = opt.step
        self.file_list = []
        self.file_list = os.listdir(opt.dir_test_images+'BR/')
        self.file_list.sort()
        self.num = len(self.file_list)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        train_data = utils.load_test(self.dir+'BR/' + self.file_list[idx])
        B_gt_data_c = utils.load_test(self.dir+'B/' + self.file_list[idx])
        R_gt_data_c = utils.load_test(self.dir+'R/' + self.file_list[idx])
        psv = torch.from_numpy(train_data).cuda()
        psv = utils.view_warp_different_fast_cuda(psv, self.min, self.max, self.step)

        return train_data, B_gt_data_c, R_gt_data_c, self.file_list[idx], psv
    





def REAL_Test_Dataset_New(opt, device):
    train_data_list,img_path = load_real(opt.dir_test_images)
    psv_list = []
    
    for train_data in train_data_list:
        psv = train_data[np.newaxis, :, :, :, :, :]
        psv = torch.from_numpy(psv)
        psv = utils.view_warp_different_fast(psv, opt.min, opt.max, opt.step)
        psv_list.append(psv)
    return [train_data_list, [], [], img_path, psv_list]




def load_real(dir):
    train_data_list = []
    image_path_list = []
    train_files = os.listdir(dir)
    train_files.sort()
    for i in range(len(train_files)):
        path = dir+train_files[i]
        tmp = cv2.imread(path + '/%d_%d.png' % (1, 1), cv2.IMREAD_COLOR)
        h ,w,c = tmp.shape
        train_data = np.zeros((7, 7, h, w, c))
        for j in range(7):
            for k in range(7):
                train_data[j,k,:,:,:] = cv2.imread(path + '/%d_%d.png' % (j+1, k+1), cv2.IMREAD_COLOR) / 255.0
        train_data = train_data[:,:,:,:,::-1]
        train_data_list.append(np.float32(train_data))
    return train_data_list, train_files