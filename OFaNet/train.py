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
from flownet import flownet
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
#scio.savemat(result_path, {'meas': meas.cpu().numpy(), 'meas_re': meas_re.cpu().numpy()})
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = './train_data/data_with_background/' # traning measurement and gt from DAVIS2017
mask_path = './train_data/data_with_background/256_simu_mask10.mat' # mask of 10 compression rate
test_path = "./test_data/meas/hike_hockey.mat"
test_ground_truth = "./test_data/gt/hike_hockey.mat"
test_aug_path = "./test_data/meas_augment/hike_hockey.mat"

mask, mask_s = generate_masks(mask_path)

last_train = 0
last_train_cnn = 0

model_save_filename = 'model-simple'
max_iter = 100
batch_size = 2
learning_rate = 0.0003
mode = 'train'  # train or test

compression_rate = 10

dataset = Imgdataset(data_path)

train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

dual_first_frame_net = dual_cnn().cuda()
rnn1 = forward_rnn().cuda()
flownet = flownet().cuda()

dict = torch.load("FlowNet2_checkpoint.pth.tar")
flownet.model.load_state_dict(dict["state_dict"])

if last_train != 0:
    dual_first_frame_net = torch.load(
        "./model-simple/.../first_frame_dual_net_model_epoch_{}.pth".format(last_train))
    flownet = torch.load("./model-simple/.../flownet_epoch_{}.pth".format(last_train))
    rnn1 = torch.load("./model-simple/.../rnn1_model_epoch_{}.pth".format(last_train))
if last_train_cnn != 0 and last_train == 0: 
    dual_first_frame_net = torch.load(
        "./model-simple/.../first_frame_dual_net_model_epoch_{}.pth".format(last_train_cnn))

loss = nn.MSELoss()
loss.cuda()

def test_simudata(test_path, epoch, result_path):
    meas = scio.loadmat(test_path)
    meas = torch.unsqueeze(torch.from_numpy(meas['meas'][:,:] / 255/10),0)
    meas = meas.cuda()
    meas = meas.float()
    meas_re = torch.div(meas*compression_rate, mask_s)
    meas_re = torch.unsqueeze(meas_re, 1)

    gt = scio.loadmat(test_ground_truth)
    gt = torch.from_numpy(gt['patch_save'] / 255)
    gt = gt.float()
    gt = gt.permute(2, 0, 1)

    aug = scio.loadmat(test_aug_path)
    aug = torch.from_numpy(aug['augment'])
    aug = aug.permute(2, 0, 1)
    aug = aug.cuda()
    aug = aug.float().unsqueeze(0)
    with torch.no_grad():
        x1_t, x2_t = dual_first_frame_net(meas, mask, 1, meas_re, aug)
        flow1 = flownet(x1_t * 255)
        model_out1 = rnn1(x1_t, aug, mask[:compression_rate, :, :], 1, mode, meas_re, flow1)
        flow2 = flownet(x2_t * 255)
        model_out2 = rnn1(x2_t, aug, mask[:compression_rate, :, :], 1, mode, meas_re, flow2)
        model_out1 = model_out1.cpu()
        model_out2 = model_out2.cpu()
        psnr1_1, psnr1_2, psnr2_1, psnr2_2 = [0,0,0,0]
        for i in range(10):
            mse1_1 = loss(x1_t[0,i,:,:].cpu() * 255, gt[i,:,:] * 255).data
            mse1_2 = loss(model_out1[0, i, :, :].cpu() * 255, gt[i, :, :] * 255).data
            psnr1_1 += 10 * torch.log10(255 * 255 / mse1_1)
            psnr1_2 += 10 * torch.log10(255 * 255 / mse1_2)
            mse2_1 = loss(x2_t[0,i,:,:].cpu() * 255, gt[i+compression_rate,:,:] * 255).data
            mse2_2 = loss(model_out2[0, i, :, :].cpu() * 255, gt[i+compression_rate, :, :] * 255).data
            psnr2_1 += 10 * torch.log10(255 * 255 / mse2_1)
            psnr2_2 += 10 * torch.log10(255 * 255 / mse2_2)
        psnr = [psnr1_1.numpy()/10, psnr1_2.numpy()/10, psnr2_1.numpy()/10, psnr2_2.numpy()/10]
        print(psnr)
        result_out_path = './' + result_path + '/' + "first_frame_dual_net_result_and_rnn_epoch_{}.mat".format(epoch)
        scio.savemat(result_out_path, {'pic1_cnn': x1_t[0,:,:,:].detach().cpu().numpy(),
                                       'model_out1':model_out1[0,:,:,:].detach().cpu().numpy(),
                                       'pic2_cnn': x2_t[0, :, :, :].detach().cpu().numpy(),
                                       'model_out2': model_out2[0, :, :, :].detach().cpu().numpy(),
                                       'PSNR': psnr})

def train(epoch, learning_rate):
    epoch_loss = 0
    begin = time.time()

    optimizer = optim.Adam([{'params': dual_first_frame_net.parameters()},
                            {'params': flownet.parameters(), 'lr':learning_rate/10},
                            {'params': rnn1.parameters()}
                            ], lr=learning_rate)

    if __name__ == '__main__':
        for iteration, batch in enumerate(train_data_loader):
            print('training:',iteration)
            gt, meas, aug = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            gt = gt.cuda()  # [batch,20,256,256]
            gt = gt.float()
            meas = meas.cuda()  # [batch,256 256]
            meas = meas.float()
            aug = aug.cuda()  # [batch,256 256]
            aug = aug.float()

            meas_re = torch.div(meas*compression_rate, mask_s)
            meas_re = torch.unsqueeze(meas_re, 1)

            optimizer.zero_grad()

            batch_size1 = gt.shape[0]
            # plt.imshow(meas.cpu().numpy()[0,:,:])
            # plt.imshow(meas_re.cpu().numpy()[0,:,:])
            x1_t, x2_t = dual_first_frame_net(meas, mask, batch_size1, meas_re, aug)

            flow1 = flownet(x1_t*255)
            model_out1 = rnn1(x1_t, aug, mask[:compression_rate,:,:], batch_size1, mode, meas_re,flow1)

            flow2 = flownet(x2_t*255)
            model_out2 = rnn1(x2_t, aug, mask[:compression_rate,:,:], batch_size1, mode, meas_re,flow2)

            Loss_cnn = loss(x1_t,gt[:,:compression_rate,:,:]) + loss(x2_t,gt[:,compression_rate:,:,:])
            Loss_rnn = loss(model_out1,gt[:,:compression_rate,:,:]) + loss(model_out2,gt[:,compression_rate:,:,:])
            Loss = Loss_cnn + Loss_rnn
            epoch_loss += Loss.data
            Loss.backward()
            optimizer.step()

    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))


def dual_checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "first_frame_dual_net_model_epoch_{}.pth".format(epoch)
    torch.save(dual_first_frame_net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def flow_checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "flownet_epoch_{}.pth".format(epoch)
    torch.save(flownet, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def rnn1_checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "rnn1_model_epoch_{}.pth".format(epoch)
    torch.save(rnn1, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def main(learning_rate):

    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon-simple' + '/' + date_time 
    model_path = 'model-simple' + '/' + date_time 
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for epoch in range(last_train + 1, last_train + max_iter + 1):
        if (epoch % 5 == 0 or epoch > 70):
            print('testing')
            dual_checkpoint(epoch, model_path)
            flow_checkpoint(epoch, model_path)
            rnn1_checkpoint(epoch, model_path)
            test_simudata(test_path, epoch, result_path)

        print('training epoch:',epoch)
        train(epoch, learning_rate)

        if (epoch % 5 == 0) and (epoch < 150):
            learning_rate = learning_rate * 0.95
            print(learning_rate)


if __name__ == '__main__':
    main(learning_rate)
