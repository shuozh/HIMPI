import os
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
import h5py
import math

def listdir_nohidden(path, type=False):
    list_nohidden = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            if type == False:
                list_nohidden.append(f)
            elif os.path.splitext(f)[1] == type:
                list_nohidden.append(f)
    return list_nohidden


def load_B_R_gt(B_dir, R_dir):
    if R_dir == None:
        B_gt_list = []
        B_list = listdir_nohidden(B_dir, type=False)
        B_list.sort()
        B_num = len(B_list)
        for i in range(B_num):
            img = load_img(B_dir + B_list[i])
            if img.max() < 0.15:
                print("Invalid file")
                continue
            B_gt_list.append(img)
            # if i == 3:
            #      break
        return B_gt_list
    else:
        R_gt_list = []
        B_gt_list = []
        B_list = listdir_nohidden(B_dir, type=False)
        B_list.sort()
        B_num = len(B_list)
        R_list = listdir_nohidden(R_dir, type=False)
        R_list.sort()
        R_num = len(R_list)
        for i in range(B_num):
            img = load_img(B_dir + B_list[i])
            if img.max() < 0.15:
                print("Invalid file")
                continue
            B_gt_list.append(img)

        for i in range(R_num):
            img = load_img(R_dir + R_list[i])
            if img.max() < 0.15:
                print("Invalid file")
                continue
            R_gt_list.append(img)
        return B_gt_list, R_gt_list



def create_B_R(B_img, R_img, crop_size=256, disparity=None):
    if disparity is not None:
        R_disparity = np.random.uniform(-disparity, -disparity / 2)
        R_img = view_warp(R_img, R_disparity)
        border = np.int(np.abs(np.floor(R_disparity)) * R_img.shape[0])
        R_img = R_img[:, :, border:-border, border:-border, :]

        B_disparity = np.random.uniform(disparity / 2, disparity)
        B_img = view_warp(B_img, B_disparity)
        border = np.int(np.abs(np.ceil(B_disparity)) * B_img.shape[0])
        B_img = B_img[:, :, border:-border, border:-border, :]

    B_img, _, _ = random_crop(B_img, crop_size)
    R_img, _, _ = random_crop(R_img, crop_size)

    factor = np.random.uniform(0.2, 0.8)
    img = B_img * factor + R_img * (1 - factor)
    u, v, h, w, c = img.shape

    return img / 255.0, B_img/ 255.0,  R_img / 255.0, factor


def create_F(lf, min, max, step):
    focus = focus_img(lf, min, max, step)
    return focus


def load_img(dir):
    img_dirs = listdir_nohidden(dir, type=False)
    img_dirs.sort()
    num = np.int(np.sqrt(len(img_dirs)))
    h, w, c = cv2.imread(dir + '/' + img_dirs[0], cv2.IMREAD_COLOR).shape
    img = np.zeros((num, num, h, w, c))

    for i in range(num):
        for j in range(num):
            index = i * num + j
            img[i, j, :, :, :] = cv2.imread(dir + '/' + img_dirs[index], cv2.IMREAD_COLOR)
            img[i, j, :, :, :] = img[i, j, :, :, ::-1]
    return np.float32(img[:,:,50:-5, 50:-5, :])


def save_img(data, dir):
    u, v, h, w, c = data.shape
    num = 0
    for i in range(u):
        for j in range(v):
            cv2.imwrite(dir + '/' + '%03d' % num + '.png', data[i, j, :, :, ::-1])
            num = num + 1


def random_crop(data, crop_hsize, crop_wsize=None, top=None, left=None):
    if crop_wsize == None:
        crop_wsize = crop_hsize
    height, width = data.shape[-3], data.shape[-2]
    if top == None:
        top = np.random.randint(0, height - crop_hsize + 1)
    if left == None:
        left = np.random.randint(0, width - crop_wsize + 1)
    data_crop = data[..., top: top + crop_hsize, left: left + crop_wsize, :]
    return data_crop, top, left


def random_flip(data, random_tmp=None):
    if random_tmp == None:
        random_tmp = np.random.random()
    if len(data.shape) == 6:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 3)
            data = np.flip(data, 1)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 4)
            data = np.flip(data, 2)
    if len(data.shape) == 5:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 2)
            data = np.flip(data, 0)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 3)
            data = np.flip(data, 1)
    if len(data.shape) == 4:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 1)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 2)
    if len(data.shape) == 3:
        if random_tmp >= (2.0 / 3):
            data = np.flip(data, 0)
        elif random_tmp <= (1.0 / 3):
            data = np.flip(data, 1)
    return data, random_tmp


def random_rotation(data, random_tmp=None):
    if random_tmp == None:
        random_tmp = np.random.choice(range(4))
    if len(data.shape) == 7:
        data = np.rot90(data, random_tmp, (2, 3))
        data = np.rot90(data, random_tmp, (4, 5))
    if len(data.shape) == 6:
        data = np.rot90(data, random_tmp, (1, 2))
        data = np.rot90(data, random_tmp, (3, 4))
    if len(data.shape) == 5:
        data = np.rot90(data, random_tmp, (0, 1))
        data = np.rot90(data, random_tmp, (2, 3))
    if len(data.shape) == 4:
        data = np.rot90(data, random_tmp, (1, 2))
    if len(data.shape) == 3:
        data = np.rot90(data, random_tmp, (0, 1))

    return data, random_tmp


def view_warp(lf, disparity):
    # warp LF with one specific disparity
    [ang_height, ang_width, height, width, channel] = lf.shape

    warped_lf = np.zeros((ang_height, ang_width, height, width, channel), dtype=np.float32)
    center_u = ang_height // 2
    center_v = ang_width // 2
    ang_u_move = list(range(-center_u, center_u + 1))
    ang_v_move = list(range(-center_v, center_v + 1))

    for u in range(ang_height):
        for v in range(ang_width):
            view = lf[u, v, :, :, :]
            u_move, v_move = ang_u_move[u] * disparity, ang_v_move[v] * disparity
            M = np.float32([[1, 0, v_move], [0, 1, u_move]])
            warped_view = cv2.warpAffine(view, M, (width, height))
            warped_lf[u, v, :, :, :] = warped_view

    return warped_lf


def view_warp_different(lf, min=-1, max=1, step=1):
    [height_view, width_view, height, width, channel] = lf.shape
    # disparity_max = disparity_num * step
    # # disparity_list = range(-disparity_max, disparity_max + step, step)
    # disparity_list = range(-disparity_max, 0, step)
    disparity_list = np.arange(min, max + step, step)
    warped_LF = np.zeros((len(disparity_list), height_view, width_view, height, width, channel), dtype=np.float32)
    for i, disparity in enumerate(disparity_list):
        warped_LF[i, :, :, :, :, :] = view_warp(lf, disparity)
    return warped_LF


def view_warp_fast(lf, disparity):
    # warp LF with one specific disparity
    [batch, height_view, width_view, height, width, channel] = lf.shape

    lf_t = lf.reshape((-1, height, width, channel)).permute((0, 3, 1, 2))  # batch*u*v,c,h,w

    center_u = height_view // 2
    center_v = width_view // 2
    grid = []
    hh = torch.arange(0, height).view(1, height, 1).expand(batch, height, width)
    ww = torch.arange(0, width).view(1, 1, width).expand(batch, height, width)
    for u in range(height_view):
        for v in range(width_view):
            dispmap_u = -disparity * (u - center_u)
            dispmap_v = -disparity * (v - center_v)
            h_range = hh + dispmap_u
            w_range = ww + dispmap_v
            h_range = 2. * h_range / (height - 1) - 1
            w_range = 2. * w_range / (width - 1) - 1
            grid_t = torch.stack((w_range, h_range), dim=3)  # [batch,h,w,2]
            grid.append(grid_t)
    grid = torch.cat(grid, 0)  # [batch*u*v,h,w,2]

    warped_lf = F.grid_sample(lf_t, grid.type_as(lf_t), 'bilinear', padding_mode="zeros", align_corners=False)
    warped_lf = warped_lf.reshape((batch, height_view, width_view, channel, height, width)).permute(
        (0, 1, 2, 4, 5, 3))

    return warped_lf

def view_warp_fast_cuda(lf, disparity):
    # warp LF with one specific disparity
    [height_view, width_view, height, width, channel] = lf.shape
    batch = 1

    lf_t = lf.reshape((-1, height, width, channel)).permute((0, 3, 1, 2))  # batch*u*v,c,h,w

    center_u = height_view // 2
    center_v = width_view // 2
    grid = []
    hh = torch.arange(0, height).view(1, height, 1).expand(batch, height, width).cuda()
    ww = torch.arange(0, width).view(1, 1, width).expand(batch, height, width).cuda()
    for u in range(height_view):
        for v in range(width_view):
            dispmap_u = -disparity * (u - center_u)
            dispmap_v = -disparity * (v - center_v)
            h_range = hh + dispmap_u
            w_range = ww + dispmap_v
            h_range = 2. * h_range / (height - 1) - 1
            w_range = 2. * w_range / (width - 1) - 1
            grid_t = torch.stack((w_range, h_range), dim=3)  # [batch,h,w,2]
            grid.append(grid_t)
    grid = torch.cat(grid, 0)  # [batch*u*v,h,w,2]

    warped_lf = F.grid_sample(lf_t, grid.type_as(lf_t), 'bilinear', padding_mode="zeros", align_corners=False)
    warped_lf = warped_lf.reshape((height_view, width_view, channel, height, width)).permute(
        (0, 1, 3, 4, 2))

    return warped_lf

def view_warp_different_fast(lf, min=-1, max=1, step=1):
    # warp LF with one specific disparity
    [batch,height_view, width_view, height, width, channel] = lf.shape
    disparity_list = np.arange(min, max + step, step)
    warped_LF = torch.zeros((batch, len(disparity_list), height_view, width_view, height, width, channel)).type_as(lf)
    for i, disparity in enumerate(disparity_list):
        disparity = round(disparity, 2)
        warped_LF[:, i, :, :, :, :, :] = view_warp_fast(lf, disparity)
    return warped_LF

def view_warp_different_fast_cuda(lf,min=-1, max=1, step=1):
    # warp LF with one specific disparity no batch

    [height_view, width_view, height, width, channel] = lf.shape
    disparity_list = np.arange(min, max + step, step)
    warped_LF = torch.zeros((len(disparity_list), height_view, width_view, height, width, channel)).type_as(lf)
    for i, disparity in enumerate(disparity_list):
        disparity = round(disparity, 2)
        warped_LF[i, :, :, :, :, :] = view_warp_fast_cuda(lf, disparity)
    return warped_LF

def focus_img(lf, min, max, step):
    if len(lf.shape) == 5:
        lf = torch.from_numpy(lf[np.newaxis])
        warped_LF = view_warp_different_fast(lf, min, max, step)
        focused_img = warped_LF.mean(axis=2).mean(axis=2)
        focused_img = np.array(focused_img[0])
    else:
        warped_LF = view_warp_different_fast(lf, min, max, step)
        focused_img = warped_LF.mean(axis=2).mean(axis=2)
    return focused_img


def load_h5(dir):
    lf = h5py.File(dir+'.h5')
    img = lf['LF']

    dir = dir.replace('h5','DL-test')
    B_img = cv2.imread(dir + '/scene1_center.png', cv2.IMREAD_COLOR) / 255.0
    B_img = B_img[:, :, ::-1]
    R_img = cv2.imread(dir + '/scene2_center.png', cv2.IMREAD_COLOR) / 255.0
    R_img = R_img[:, :, ::-1]
    return np.float32(img), np.float32(B_img), np.float32(R_img)

def load_test(dir):
    img_dirs = listdir_nohidden(dir, type=False)
    if 'B.png' in img_dirs:
        img_dirs.remove('B.png')
        img_dirs.remove('R.png')
        img_dirs.sort()
        num = np.int(np.sqrt(len(img_dirs)))
        h, w, c = cv2.imread(dir + '/' + img_dirs[0], cv2.IMREAD_COLOR).shape
        img = np.zeros((num, num, h, w, c))
        for i in range(num):
            for j in range(num):
                index = i * num + j
                img[i, j, :, :, :] = cv2.imread(dir + '/' + img_dirs[index], cv2.IMREAD_COLOR) / 255.0
                img[i, j, :, :, :] = img[i, j, :, :, ::-1]
        img = img[0:7,0:7]
        B_img = cv2.imread(dir + '/B.png', cv2.IMREAD_COLOR) / 255.0
        B_img = B_img[:, :, ::-1]
        R_img = cv2.imread(dir + '/R.png', cv2.IMREAD_COLOR) / 255.0
        R_img = R_img[:, :, ::-1]
        return np.float32(img), np.float32(B_img), np.float32(R_img)
    else:
        img_dirs.sort()
        num = np.int(np.sqrt(len(img_dirs)))
        h, w, c = cv2.imread(dir + '/' + img_dirs[0], cv2.IMREAD_COLOR).shape
        img = np.zeros((num, num, h, w, c))
        for i in range(num):
            for j in range(num):
                index = i * num + j
                img[i, j, :, :, :] = cv2.imread(dir + '/' + img_dirs[index], cv2.IMREAD_COLOR) / 255.0
                img[i, j, :, :, :] = img[i, j, :, :, ::-1]
        img = img[0:7,0:7]
        return np.float32(img)


def load_test_full(dir):
    train_data_list = []
    B_gt_data_list = []
    R_gt_data_list = []
    image_path_list = []
    files = os.listdir(dir)
    # files.remove('opt.txt')
    files.sort()
    for image_path in files:
        train_data, B_gt_data, R_gt_data = load_test(dir + image_path)
        train_data_list.append(train_data)
        B_gt_data_list.append(B_gt_data)
        R_gt_data_list.append(R_gt_data)
        image_path_list.append(image_path)

    return [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list]


def load_test_new(dir):
    train_data_list = []
    B_gt_data_list = []
    R_gt_data_list = []
    image_path_list = []
    train_files = os.listdir(dir+'BR/')
    B_files = os.listdir(dir+'B/')
    R_files = os.listdir(dir+'R/')
    train_files.sort()
    B_files.sort()
    R_files.sort()
    # files.remove('opt.txt')
    for image_path in train_files:
        train_data = load_test(dir+'BR/' + image_path)
        train_data_list.append(train_data)
        image_path_list.append(image_path)
    for image_path in B_files:
        B_gt_data = load_test(dir+'B/' + image_path)
        B_gt_data_list.append(B_gt_data)
    for image_path in R_files:
        R_gt_data = load_test(dir+'R/' + image_path)
        R_gt_data_list.append(R_gt_data)
    return [train_data_list, B_gt_data_list, R_gt_data_list, image_path_list]

def crop_view(patchsize, data, bdr, idxs=None):
    if data.shape[-1] <= 4:  #if channel dim at end
        h_dim = len(data.shape) - 3
    else:
        h_dim = len(data.shape) - 2

    h, w = data.shape[h_dim], data.shape[h_dim+1]
    patchsize_h,patchsize_w = patchsize[0],patchsize[1]
    if idxs is None:
        h_idx = np.random.randint(bdr, h - patchsize_h - bdr)
        w_idx = np.random.randint(bdr, w - patchsize_w - bdr)
        idxs=[h_idx,w_idx]
    else:
        h_idx,w_idx=idxs[0],idxs[1]

    indx = (slice(None),)*(h_dim) + (slice(h_idx,h_idx + patchsize_h),) + (slice(w_idx, w_idx + patchsize_w),) + (slice(None),)*(data.ndim-h_dim-2)
    data = data[indx]
    return data,idxs

def crop_multiview(patchsize, datas, bdr, idxs):
    if isinstance(datas,tuple):
        new_datas=()
        for i, input in enumerate(datas):
            new_data,_ = crop_view(patchsize, input, bdr, idxs)
            new_datas += (new_data,)
    else:
        new_datas,_ = crop_view(patchsize, datas, bdr, idxs)
    return new_datas

def get_parameter_number(net):
    print(net)
    parameter_list = [p.numel() for p in net.parameters()]
    print(parameter_list)
    total_num = sum(parameter_list)
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)



def result_save_all(opt,save_dir,image_path ,target_view_b,target_view_f,target_view_bs,target_view_fs, gt_target_b,gt_target_f,cob, alb,cof,alf,target_position_list):
    cob = cob.cpu().detach().numpy() * 255
    cof = cof.cpu().detach().numpy() * 255
    alb = alb.cpu().detach().numpy() * 255
    alf = alf.cpu().detach().numpy() * 255
    factor = opt.factor
    #print(target_position)
    for k in range(len(target_view_b)):
        [a,b] = target_position_list[k]
        filename = save_dir + image_path + '/' + str(a) + '_' + str(b) + '/'
        if not os.path.exists(filename):
            os.makedirs(filename)
        target_view_bf = target_view_b[k]*factor+target_view_f[k]*(1-factor)
        #gtbf = gt_target[k].cpu().detach().numpy() * 255
        gtb = gt_target_b[k].cpu().detach().numpy() * 255
        gtf = gt_target_f[k].cpu().detach().numpy() * 255
        tbf = target_view_bf.cpu().detach().numpy() * 255
        #tbfs = target_views_list[k].cpu().detach().numpy() * 255
        tb = target_view_b[k].cpu().detach().numpy() * 255
        tf = target_view_f[k].cpu().detach().numpy() * 255 
        tbs = target_view_bs[k].cpu().detach().numpy() * 255
        tfs = target_view_fs[k].cpu().detach().numpy() * 255
        # print(tb[0].shape)
        # print('gtb',gtb.shape)
        cv2.imwrite(filename+'bf'+str(k)+'.png', tbf[0][:, :, ::-1])
        #cv2.imwrite(filename+'bfs'+str(k)+'.png', tbfs[j][:, :, ::-1])
        # cv2.imwrite(filename+'bf_gt'+str(k)+'.png', gtbf[j][:, :, ::-1])
        cv2.imwrite(filename+'b'+str(k)+'.png', tb[0][:, :, ::-1])
        cv2.imwrite(filename+'bs'+str(k)+'.png', tbs[0][:, :, ::-1])
        cv2.imwrite(filename+'b_gt'+str(k)+'.png', gtb[:, :, ::-1])
        cv2.imwrite(filename+'f_gt'+str(k)+'.png', gtf[:, :, ::-1])
        cv2.imwrite(filename+'f'+str(k)+'.png', tf[0][:, :, ::-1])
        cv2.imwrite(filename+'fs'+str(k)+'.png', tfs[0][:, :, ::-1])
    filename = save_dir + image_path + '/mpi/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    for h in range(cob.shape[1]):
        cv2.imwrite(filename+'color_b_'+str(h)+'.png', cob[0][h].transpose(1,2,0)[:, :, ::-1] )
        cv2.imwrite(filename+'color_f_'+str(h)+'.png', cof[0][h].transpose(1,2,0)[:, :, ::-1] )
        cv2.imwrite(filename+'alpha_b_'+str(h)+'.png', alb[0][h].transpose(1,2,0))
        cv2.imwrite(filename+'alpha_f_'+str(h)+'.png', alf[0][h].transpose(1,2,0))


def result_save_all1(opt,save_dir,image_path ,target_view_b,target_view_f,target_view_bs,target_view_fs, gt_target_b,gt_target_f,cob, alb,cof,alf,target_position_list):
    cob = cob.cpu().detach().numpy() * 255
    cof = cof.cpu().detach().numpy() * 255
    alb = alb.cpu().detach().numpy() * 255
    alf = alf.cpu().detach().numpy() * 255
    factor = opt.factor
    #print(target_position)
    for k in range(len(target_view_b)):
        [a,b] = target_position_list[k]
        filename = save_dir + image_path + '/' + str(a) + '_' + str(b) + '/'
        if not os.path.exists(filename):
            os.makedirs(filename)
        target_view_bf = target_view_b[k]*factor+target_view_f[k]*(1-factor)
        #gtbf = gt_target[k].cpu().detach().numpy() * 255
        gtb = gt_target_b[k] * 255
        gtf = gt_target_f[k] * 255
        tbf = target_view_bf.cpu().detach().numpy() * 255
        #tbfs = target_views_list[k].cpu().detach().numpy() * 255
        tb = target_view_b[k].cpu().detach().numpy()* 255
        tf = target_view_f[k].cpu().detach().numpy()* 255 
        tbs = target_view_bs[k].cpu().detach().numpy() * 255
        tfs = target_view_fs[k].cpu().detach().numpy()* 255
        # print(tb[0].shape)
        # print('gtb',gtb.shape)
        cv2.imwrite(filename+'bf'+str(k)+'.png', tbf[0][:, :, ::-1])
        #cv2.imwrite(filename+'bfs'+str(k)+'.png', tbfs[j][:, :, ::-1])
        # cv2.imwrite(filename+'bf_gt'+str(k)+'.png', gtbf[j][:, :, ::-1])
        cv2.imwrite(filename+'b'+str(k)+'.png', tb[0][:, :, ::-1])
        cv2.imwrite(filename+'bs'+str(k)+'.png', tbs[0][:, :, ::-1])
        cv2.imwrite(filename+'b_gt'+str(k)+'.png', gtb[:, :, ::-1])
        cv2.imwrite(filename+'f_gt'+str(k)+'.png', gtf[:, :, ::-1])
        cv2.imwrite(filename+'f'+str(k)+'.png', tf[0][:, :, ::-1])
        cv2.imwrite(filename+'fs'+str(k)+'.png', tfs[0][:, :, ::-1])
    filename = save_dir + image_path + '/mpi/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    for h in range(cob.shape[1]):
        cv2.imwrite(filename+'color_b_'+str(h)+'.png', cob[0][h].transpose(1,2,0)[:, :, ::-1] )
        cv2.imwrite(filename+'color_f_'+str(h)+'.png', cof[0][h].transpose(1,2,0)[:, :, ::-1] )
        cv2.imwrite(filename+'alpha_b_'+str(h)+'.png', alb[0][h].transpose(1,2,0))
        cv2.imwrite(filename+'alpha_f_'+str(h)+'.png', alf[0][h].transpose(1,2,0))



    cob = cob.cpu().detach().numpy() * 255
    cof = cof.cpu().detach().numpy() * 255
    alb = alb.cpu().detach().numpy() * 255
    alf = alf.cpu().detach().numpy() * 255
    factor = opt.factor
    #print(target_position)
    for k in range(len(target_view_b)):
        [a,b] = target_position_list[k]
        filename = save_dir + image_path + '/' + str(a) + '_' + str(b) + '/'
        if not os.path.exists(filename):
            os.makedirs(filename)
        #target_view_bf = target_view_b[k]*factor+target_view_f[k]*(1-factor)
        #gtbf = gt_target[k].cpu().detach().numpy() * 255
        # gtb = gt_target_b[k] * 255
        # gtf = gt_target_f[k] * 255
        # tbf = target_view_bf.cpu().detach().numpy() * 255
        #tbfs = target_views_list[k].cpu().detach().numpy() * 255
        tb = target_view_b[k].cpu().detach().numpy()* 255
        tf = target_view_f[k].cpu().detach().numpy()* 255 
        # tbs = target_view_bs[k].cpu().detach().numpy() * 255
        # tfs = target_view_fs[k].cpu().detach().numpy()* 255
        # print(tb[0].shape)
        # print('gtb',gtb.shape)
        # cv2.imwrite(filename+'bf'+str(k)+'.png', tbf[0][:, :, ::-1])
        # #cv2.imwrite(filename+'bfs'+str(k)+'.png', tbfs[j][:, :, ::-1])
        # # cv2.imwrite(filename+'bf_gt'+str(k)+'.png', gtbf[j][:, :, ::-1])
        cv2.imwrite(filename+'b'+str(k)+'.png', tb[0][:, :, ::-1])
        # cv2.imwrite(filename+'bs'+str(k)+'.png', tbs[0][:, :, ::-1])
        # cv2.imwrite(filename+'b_gt'+str(k)+'.png', gtb[:, :, ::-1])
        # cv2.imwrite(filename+'f_gt'+str(k)+'.png', gtf[:, :, ::-1])
        cv2.imwrite(filename+'f'+str(k)+'.png', tf[0][:, :, ::-1])
        # cv2.imwrite(filename+'fs'+str(k)+'.png', tfs[0][:, :, ::-1])
    filename = save_dir + image_path + '/mpi/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    for h in range(cob.shape[1]):
        cv2.imwrite(filename+'color_b_'+str(h)+'.png', cob[0][h].transpose(1,2,0)[:, :, ::-1] )
        cv2.imwrite(filename+'color_f_'+str(h)+'.png', cof[0][h].transpose(1,2,0)[:, :, ::-1] )
        cv2.imwrite(filename+'alpha_b_'+str(h)+'.png', alb[0][h].transpose(1,2,0))
        cv2.imwrite(filename+'alpha_f_'+str(h)+'.png', alf[0][h].transpose(1,2,0))


def get_target_view(gt_data,gt_b,gt_f,target_position_list):
    gt_target_list = []
    gt_target_b_list = []
    gt_target_f_list = []
    for [u,v] in target_position_list:
        gt_target = gt_data[...,u,v,:,:,:]
        gt_target_b = gt_b[...,u,v,:,:,:]#[:,:,w * (v):w * (v+1), h * (u):h * (u+1)]
        gt_target_f = gt_f[...,u,v,:,:,:]
        #gt_target = einops.rearrange(gt_target, 'b c w h -> b w h c')
        #gt_target_b = einops.rearrange(gt_target_b, 'b c w h -> b w h c')
        gt_target_list.append(gt_target.cuda())
        gt_target_b_list.append(gt_target_b.cuda())
        gt_target_f_list.append(gt_target_f.cuda())
    #gt_target_list = torch.cat(gt_target_list, 0)
    #gt_target_b_list = torch.cat(gt_target_b_list, 0)#[:,:, w * (an // 2):w * (an // 2 + 1),h * (an // 2):h * (an // 2 + 1)]
    #mid_train_data = einops.rearrange(mid_train_data, 'b c w h -> b w h c')
    return  gt_target_list,  gt_target_b_list, gt_target_f_list

def get_target_view_n(gt_data,gt_b,gt_f,target_position_list):
    gt_target_list = []
    gt_target_b_list = []
    gt_target_f_list = []
    for [u,v] in target_position_list:
        gt_target = gt_data[u,v,:,:,:]
        gt_target_b = gt_b[u,v,:,:,:]#[:,:,w * (v):w * (v+1), h * (u):h * (u+1)]
        gt_target_f = gt_f[u,v,:,:,:]
        #gt_target = einops.rearrange(gt_target, 'b c w h -> b w h c')
        #gt_target_b = einops.rearrange(gt_target_b, 'b c w h -> b w h c')
        gt_target_list.append(gt_target)
        gt_target_b_list.append(gt_target_b)
        gt_target_f_list.append(gt_target_f)
    #gt_target_list = torch.cat(gt_target_list, 0)
    #gt_target_b_list = torch.cat(gt_target_b_list, 0)#[:,:, w * (an // 2):w * (an // 2 + 1),h * (an // 2):h * (an // 2 + 1)]
    #mid_train_data = einops.rearrange(mid_train_data, 'b c w h -> b w h c')
    return  gt_target_list,  gt_target_b_list, gt_target_f_list

def test_crop(x, model, view_n_new, target_position_list, disparity_list, scale=1, max_length=48, shave=0, mod=1):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    b, d, u, v, h, w, c = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = int(math.ceil((h_half + shave) / mod) * mod), int(math.ceil((w_half + shave) / mod) * mod)
    lr_list = [
        x[..., 0:h_size, 0:w_size, :],
        x[..., 0:h_size, (w - w_size):w, :],
        x[..., (h - h_size):h, 0:w_size, :],
        x[..., (h - h_size):h, (w - w_size):w, :]]

    sr_list_b = []
    sr_list_f = []
    sr_list_bs = []
    sr_list_fs = []
    for i in range(view_n_new* view_n_new):
        sr_list_b.append([])
        sr_list_f.append([])
        sr_list_bs.append([])
        sr_list_fs.append([])
    co_list_b = []
    co_list_f = []
    al_list_b = []
    al_list_f = []
    for i in range(0, 4, n_GPUs):
        lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
        if lr_batch.shape[3] > max_length or lr_batch.shape[4] > max_length:
            target_view_b, target_view_f, color_b, alpha_b, color_f, alpha_f, \
                target_view_list_b_s,target_view_list_f_s = test_crop(lr_batch, model, view_n_new, target_position_list, disparity_list,
                                                 scale=scale, max_length=max_length, shave=shave, mod=mod)
        else:
            target_view_b, target_view_f,color_b, alpha_b, color_f, alpha_f,\
                 target_view_list_b_s,target_view_list_f_s = model(lr_batch, target_position_list=target_position_list, view_n_new=view_n_new, disparity_list=disparity_list)
        for k in range(len(target_view_b)): 
            sr_list_b[k].extend(target_view_b[k].chunk(n_GPUs, dim=0))
            sr_list_f[k].extend(target_view_f[k].chunk(n_GPUs, dim=0))
            sr_list_bs[k].extend(target_view_list_b_s[k].chunk(n_GPUs, dim=0))
            sr_list_fs[k].extend(target_view_list_f_s[k].chunk(n_GPUs, dim=0))

        co_list_b.extend(color_b.chunk(n_GPUs, dim=0))
        al_list_b.extend(alpha_b.chunk(n_GPUs, dim=0)) 
        co_list_f.extend(color_f.chunk(n_GPUs, dim=0))
        al_list_f.extend(alpha_f.chunk(n_GPUs, dim=0))

    h, w = scale * h, scale * w
    output1 = []
    output2 = []
    output9 = []
    output10 = []
    for k in range(len(target_view_b)):
        output1.append(sum_pic(x, sr_list_b[k], b, 3, h, w)) 
        output2.append(sum_pic(x, sr_list_f[k], b, 3, h, w)) 
        output9.append(sum_pic(x, sr_list_bs[k], b, 3, h, w))
        output10.append( sum_pic(x, sr_list_fs[k], b, 3, h, w))
    output4 = sum_mpi(x,co_list_b,b,color_b.shape[2],h,w,color_b.shape[1])
    output5 = sum_mpi(x,al_list_b,b,alpha_b.shape[2],h,w,alpha_b.shape[1])
    output6 = sum_mpi(x,co_list_f,b,color_f.shape[2],h,w,color_f.shape[1])
    output7 = sum_mpi(x,al_list_f,b,alpha_f.shape[2],h,w,alpha_f.shape[1])

    return output1, output2 , output4, output5, output6, output7, output9, output10

def test_crop1(x, model, view_n_new, target_position_list, disparity_list, scale=1, max_length=48, shave=0, mod=1):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    b, d, u, v, h, w, c = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = int(math.ceil((h_half + shave) / mod) * mod), int(math.ceil((w_half + shave) / mod) * mod)
    lr_list = [
        x[..., 0:h_size, 0:w_size, :],
        x[..., 0:h_size, (w - w_size):w, :],
        x[..., (h - h_size):h, 0:w_size, :],
        x[..., (h - h_size):h, (w - w_size):w, :]]

    sr_list_b = []
    sr_list_f = []
    sr_list_bs = []
    sr_list_fs = []
    sr_list_bs2 = []
    sr_list_fs2 = []
    for i in range(view_n_new* view_n_new):
        sr_list_b.append([])
        sr_list_f.append([])
        sr_list_bs.append([])
        sr_list_fs.append([])
        sr_list_bs2.append([])
        sr_list_fs2.append([])
    co_list_b = []
    co_list_f = []
    al_list_b = []
    al_list_f = []
    for i in range(0, 4, n_GPUs):
        lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
        if lr_batch.shape[3] > max_length or lr_batch.shape[4] > max_length:
            target_view_b, target_view_f, color_b, alpha_b, color_f, alpha_f, \
                target_view_list_b_s,target_view_list_f_s,target_view_list_b_s2,target_view_list_f_s2 = test_crop1(lr_batch, model, view_n_new, target_position_list, disparity_list,
                                                 scale=scale, max_length=max_length, shave=shave, mod=mod)
        else:
            target_view_b, target_view_f,color_b, alpha_b, color_f, alpha_f,\
                 target_view_list_b_s,target_view_list_f_s,target_view_list_b_s2,target_view_list_f_s2 = model(lr_batch, target_position_list=target_position_list, view_n_new=view_n_new, disparity_list=disparity_list)
        for k in range(len(target_view_b)): 
            sr_list_b[k].extend(target_view_b[k].chunk(n_GPUs, dim=0))
            sr_list_f[k].extend(target_view_f[k].chunk(n_GPUs, dim=0))
            sr_list_bs[k].extend(target_view_list_b_s[k].chunk(n_GPUs, dim=0))
            sr_list_fs[k].extend(target_view_list_f_s[k].chunk(n_GPUs, dim=0))
            sr_list_bs2[k].extend(target_view_list_b_s2[k].chunk(n_GPUs, dim=0))
            sr_list_fs2[k].extend(target_view_list_f_s2[k].chunk(n_GPUs, dim=0))

        co_list_b.extend(color_b.chunk(n_GPUs, dim=0))
        al_list_b.extend(alpha_b.chunk(n_GPUs, dim=0)) 
        co_list_f.extend(color_f.chunk(n_GPUs, dim=0))
        al_list_f.extend(alpha_f.chunk(n_GPUs, dim=0))

    h, w = scale * h, scale * w
    output1 = []
    output2 = []
    output9 = []
    output10 = []
    output11 = []
    output12 = []
    for k in range(len(target_view_b)):
        output1.append(sum_pic(x, sr_list_b[k], b, 3, h, w)) 
        output2.append(sum_pic(x, sr_list_f[k], b, 3, h, w)) 
        output9.append(sum_pic(x, sr_list_bs[k], b, 3, h, w))
        output10.append( sum_pic(x, sr_list_fs[k], b, 3, h, w))
        output11.append(sum_pic(x, sr_list_bs2[k], b, 3, h, w))
        output12.append( sum_pic(x, sr_list_fs2[k], b, 3, h, w))
    output4 = sum_mpi(x,co_list_b,b,color_b.shape[2],h,w,color_b.shape[1])
    output5 = sum_mpi(x,al_list_b,b,alpha_b.shape[2],h,w,alpha_b.shape[1])
    output6 = sum_mpi(x,co_list_f,b,color_f.shape[2],h,w,color_f.shape[1])
    output7 = sum_mpi(x,al_list_f,b,alpha_f.shape[2],h,w,alpha_f.shape[1])

    return output1, output2 , output4, output5, output6, output7, output9, output10,output11, output12

def test_cropo(x, model, view_n_new, target_position_list, disparity_list, scale=1, max_length=48, shave=0, mod=1):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    b, d, u, v, h, w, c = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = int(math.ceil((h_half + shave) / mod) * mod), int(math.ceil((w_half + shave) / mod) * mod)
    lr_list = [
        x[..., 0:h_size, 0:w_size, :],
        x[..., 0:h_size, (w - w_size):w, :],
        x[..., (h - h_size):h, 0:w_size, :],
        x[..., (h - h_size):h, (w - w_size):w, :]]

    sr_list_b = []
    sr_list_f = []
    sr_list_bs = []
    sr_list_fs = []
    sr_list_bs2 = []
    sr_list_fs2 = []
    for i in range(view_n_new* view_n_new):
        sr_list_b.append([])
        sr_list_f.append([])
        sr_list_bs.append([])
        sr_list_fs.append([])
        sr_list_bs2.append([])
        sr_list_fs2.append([])
    co_list_b = []
    co_list_f = []
    al_list_b = []
    al_list_f = []
    for i in range(0, 4, n_GPUs):
        lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
        if lr_batch.shape[3] > max_length or lr_batch.shape[4] > max_length:
            target_view_b, target_view_f, color_b, alpha_b, color_f, alpha_f, \
                target_view_list_b_s,target_view_list_f_s,target_view_list_b_s2,target_view_list_f_s2 = test_cropo(lr_batch, model, view_n_new, target_position_list, disparity_list,
                                                 scale=scale, max_length=max_length, shave=shave, mod=mod)
        else:
            target_view_b, target_view_f,color_b, alpha_b, color_f, alpha_f,\
                 target_view_list_b_s,target_view_list_f_s,target_view_list_b_s2,target_view_list_f_s2 = model(lr_batch, target_position_list=target_position_list, view_n_new=view_n_new, disparity_list=disparity_list)
        for k in range(len(target_view_b)): 
            sr_list_b[k].extend(target_view_b[k].chunk(n_GPUs, dim=0))
            sr_list_f[k].extend(target_view_f[k].chunk(n_GPUs, dim=0))
            sr_list_bs[k].extend(target_view_list_b_s[k].chunk(n_GPUs, dim=0))
            sr_list_fs[k].extend(target_view_list_f_s[k].chunk(n_GPUs, dim=0))
            sr_list_bs2[k].extend(target_view_list_b_s2[k].chunk(n_GPUs, dim=0))
            sr_list_fs2[k].extend(target_view_list_f_s2[k].chunk(n_GPUs, dim=0))

        co_list_b.extend(color_b.chunk(n_GPUs, dim=0))
        al_list_b.extend(alpha_b.chunk(n_GPUs, dim=0)) 
        co_list_f.extend(color_f.chunk(n_GPUs, dim=0))
        al_list_f.extend(alpha_f.chunk(n_GPUs, dim=0))

    h, w = scale * h, scale * w
    output1 = []
    output2 = []
    output9 = []
    output10 = []
    output11 = []
    output12 = []
    for k in range(len(target_view_b)):
        output1.append(sum_pic(x, sr_list_b[k], b, 3, h, w)) 
        output2.append(sum_pic(x, sr_list_f[k], b, 3, h, w)) 
        output9.append(sum_pic(x, sr_list_bs[k], b, 3, h, w))
        output10.append( sum_pic(x, sr_list_fs[k], b, 3, h, w))
        output11.append(sum_pic(x, sr_list_bs2[k], b, 3, h, w))
        output12.append( sum_pic(x, sr_list_fs2[k], b, 3, h, w))
    output4 = sum_mpi(x,co_list_b,b,color_b.shape[2],h,w,color_b.shape[1])
    output5 = sum_mpi(x,al_list_b,b,alpha_b.shape[2],h,w,alpha_b.shape[1])
    output6 = sum_mpi(x,co_list_f,b,color_f.shape[2],h,w,color_f.shape[1])
    output7 = sum_mpi(x,al_list_f,b,alpha_f.shape[2],h,w,alpha_f.shape[1])

    return output1, output2 , output4, output5, output6, output7, output9, output10,output11, output12


def sum_pic(x, sr_list, b, c, h, w):
    h_half = h//2
    w_half = w//2
    output = x.new(b, h, w, c)
    output[:, 0:h_half, 0:w_half, :] = sr_list[0][:, :h_half, :w_half, :]
    output[:, 0:h_half, w_half:w, :] = sr_list[1][:, :h_half, -(w - w_half):, :]
    output[:, h_half:h, 0:w_half, :] = sr_list[2][:, -(h - h_half):, :w_half, :]
    output[:, h_half:h, w_half:w, :] = sr_list[3][:, -(h - h_half):, -(w - w_half):, :]
    return output

def sum_mpi(x, sr_list, b, c, h, w,d):
    h_half = h//2
    w_half = w//2
    output = x.new(b,d, c, h, w )
    output[:, :, :, 0:h_half, 0:w_half] = sr_list[0][:,:,:, :h_half, :w_half]
    output[:, :, :, 0:h_half, w_half:w] = sr_list[1][:,:,:, :h_half, -(w - w_half):]
    output[:, :, :, h_half:h, 0:w_half] = sr_list[2][:,:,:, -(h - h_half):, :w_half]
    output[:, :, :, h_half:h, w_half:w] = sr_list[3][:,:,:, -(h - h_half):, -(w - w_half):]
    return output