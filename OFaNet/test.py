"""
@author : RuiyingLu, Bo Chen
@Email : ruiyinglu_xidian@163.com      bchen@mail.xidian.edu.cn

LICENSE
=======================================================================

The code is for research purpose only. All rights reserved.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

"""

from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import dual_cnn, forward_rnn, cnn1, backrnn
from util import generate_masks, time2file_name
import torch.optim as optim
from flownet import flownet
import torch.nn as nn
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import datetime
import os
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

mask_path = './test_data/256_simu_mask10.mat' # mask of 10 compression rate

test_path = "./test_data"


mask, mask_s = generate_masks(mask_path)

last_train = 0
max_iter = 100
batch_size = 2
learning_rate = 0.0003
mode = 'train'  # train or test

compression_rate = 10

dual_first_frame_net = dual_cnn().cuda()
flownet = flownet().cuda()
rnn1 = forward_rnn().cuda()

loss = nn.MSELoss()
loss.cuda()


def test_simudata(test_gt, test_meas, test_aug, name, epoch, result_path, std):
    meas = scio.loadmat(test_meas)
    meas = torch.unsqueeze(torch.from_numpy(meas['meas'] / 255/10),0)
    meas = meas.cuda()
    meas = meas.float()
    noise = torch.randn(meas.shape) * std    # * torch.max(meas)
    meas = meas+noise.cuda()
    meas_re = torch.div(meas*compression_rate, mask_s)
    meas_re = torch.unsqueeze(meas_re, 1)

    gt = scio.loadmat(test_gt)
    gt = torch.from_numpy(gt['patch_save'] / 255)
    gt = gt.float()
    gt = gt.permute(2, 0, 1)

    aug = scio.loadmat(test_aug)
    aug = torch.from_numpy(aug['augment'])
    aug = aug.permute(2, 0, 1)
    aug = aug.cuda()
    aug = aug.float().unsqueeze(0)

    dual_first_frame_net = torch.load( "./model-simple/.../first_frame_dual_net_model_epoch_{}.pth".format(epoch))
    flownet = torch.load("./model-simple/.../flownet_epoch_{}.pth".format(epoch))
    rnn1 = torch.load( "./model-simple/.../rnn1_model_epoch_{}.pth".format(epoch))

    with torch.no_grad():
        time1 = datetime.datetime.now()
        x1_t, x2_t = dual_first_frame_net(meas, mask, 1, meas_re, aug)
        time2 = datetime.datetime.now()
        print('time:',time2-time1)
        flow1 = flownet(x1_t * 255)
        model_out1 = rnn1(x1_t, aug, mask[:compression_rate, :, :], 1, mode, meas_re, flow1)
        flow2 = flownet(x2_t * 255)
        model_out2 = rnn1(x2_t, aug, mask[:compression_rate, :, :], 1, mode, meas_re, flow2)
        time3 = datetime.datetime.now()
        print('time:',time3-time1)
        psnr1_1, psnr1_2, psnr2_1, psnr2_2 = [0,0,0,0]
        for i in range(10):
            mse1_1 = loss(x1_t[0,i,:,:].cpu() * 255, gt[i,:,:].cpu() * 255).data
            mse1_2 = loss(model_out1[0, i, :, :].cpu() * 255, gt[i, :, :].cpu() * 255).data
            psnr1_1 += 10 * torch.log10(255 * 255 / mse1_1)
            psnr1_2 += 10 * torch.log10(255 * 255 / mse1_2)
            mse2_1 = loss(x2_t[0,i,:,:].cpu() * 255, gt[i+compression_rate,:,:].cpu() * 255).data
            mse2_2 = loss(model_out2[0, i, :, :].cpu() * 255, gt[i+compression_rate, :, :].cpu() * 255).data
            psnr2_1 += 10 * torch.log10(255 * 255 / mse2_1)
            psnr2_2 += 10 * torch.log10(255 * 255 / mse2_2)
        psnr = [psnr1_1.numpy()/10, psnr1_2.numpy()/10, psnr2_1.numpy()/10, psnr2_2.numpy()/10]
        print('PSNR:',psnr)  
        result_out_path = './' + result_path + '/' + name[:-4]  +"_epoch_{}.mat".format(epoch)
        scio.savemat(result_out_path, {'pic1_cnn': x1_t[0,:,:,:].detach().cpu().numpy(),
                                       'model_out1':model_out1[0,:,:,:].detach().cpu().numpy(),
                                       'pic2_cnn': x2_t[0, :, :, :].detach().cpu().numpy(),
                                       'model_out2': model_out2[0, :, :, :].detach().cpu().numpy(),
                                       'PSNR': psnr})
        return psnr

def main(learning_rate):

    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    std = 0
    test_epoch = 90
    result_path = 'epoch'+str(test_epoch)+'-Overall_results_with_noise_'+str(std)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    test_gt_path = os.listdir(test_path+'/gt')
    test_gt_path.sort()
    for epoch in range(last_train + 1, last_train + max_iter + 1):
        if (epoch % 1 == 0 and epoch <= 97 and epoch == test_epoch):
            print('testing:',epoch)
            for i in range(6):
                test_gt = test_path+'/gt/'+test_gt_path[i*2+1]
                test_meas = test_path+'/meas/'+test_gt_path[i*2+1]
                test_aug = test_path+'/meas_augment/'+test_gt_path[i*2+1]
                print("####### testing the data ##########", test_gt_path[i*2+1])
                PSNR = np.array(test_simudata(test_gt, test_meas, test_aug, test_gt_path[i*2+1], epoch, result_path, std))
                if i==0:
                    PSNR_all = PSNR
                else:
                    PSNR_all = PSNR_all + PSNR
            print("The overall average PSNR:", PSNR_all/6)


if __name__ == '__main__':
    main(learning_rate)
