"""
@author : Ziheng Cheng, Bo Chen
@Email : zhcheng@stu.xidian.edu.cn      bchen@mail.xidian.edu.cn

Description:
    This is the data generating code for Snapshot Compressive Imaging reconstruction in recurrent convolution neural network

Citation:
    The code prepares for ECCV 2020

Contact:
    Ziheng Cheng
    zhcheng@stu.xidian.edu.cn
    Xidian University, Xi'an, China

    Bo Chen
    bchen@mail.xidian.edu.cn
    Xidian University, Xi'an, China

LICENSE
=======================================================================

The code is for research purpose only. All rights reserved.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

Copyright (c), 2020, Ziheng Cheng
zhcheng@stu.xidian.edu.cn

"""

from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio


class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            dir_list = os.listdir(path)
            groung_truth_path = path + '/gt'
            measurement_path = path + '/meas'
            augment_path = path+'/meas_augment'

            if os.path.exists(groung_truth_path) and os.path.exists(measurement_path) and os.path.exists(augment_path):
                groung_truth = os.listdir(groung_truth_path)
                measurement = os.listdir(measurement_path)
                augment = os.listdir(augment_path)

                self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i],
                              'measurement': measurement_path + '/' + measurement[i],
                              'augment': augment_path + '/' + augment[i]} for i in range(len(groung_truth))] #
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        groung_truth, measurement, augment = self.data[index]["groung_truth"], self.data[index]["measurement"], self.data[index]["augment"]
        # print(groung_truth)
        gt = scio.loadmat(groung_truth)
        meas = scio.loadmat(measurement)
        aug = scio.loadmat(augment)
        if "patch_save" in gt:
            gt = torch.from_numpy(gt['patch_save'] / 255)
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)

        meas = torch.from_numpy(meas['meas'] / 255/(gt.shape[2]/2))

        aug = torch.from_numpy(aug['augment'])
        aug = aug.permute(2, 0, 1)

        gt = gt.permute(2, 0, 1)

        # print(tran(img).shape)

        return gt, meas, aug

    def __len__(self):

        return len(self.data)
