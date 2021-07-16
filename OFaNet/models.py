"""
@author : RuiyingLu, Bo Chen
@Email : ruiyinglu_xidian@163.com      bchen@mail.xidian.edu.cn

LICENSE
=======================================================================

The code is for research purpose only. All rights reserved.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

"""

from my_tools import *


class dual_cnn(nn.Module):

    def __init__(self):
        super(dual_cnn, self).__init__()
        self.conv1_1 = nn.Conv2d(15, 32, kernel_size=5, stride=1, padding=2)
        self.relu1_1 = nn.LeakyReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.LeakyReLU(inplace=True)
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.relu3_1 = nn.LeakyReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.relu4_1 = nn.LeakyReLU(inplace=True)


        self.conv5_1 = nn.ConvTranspose2d(64*4, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5_1 = nn.LeakyReLU(inplace=True)
        self.conv51_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu51_1 = nn.LeakyReLU(inplace=True)
        self.conv52_1 = nn.Conv2d(32, 16, kernel_size=1, stride=1)
        self.relu52_1 = nn.LeakyReLU(inplace=True)
        self.conv6_1 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)
        self.sigmoid6_1 = nn.Sigmoid()

        self.res_part1_1 = res_part(64, 64)
        self.res_part2_1 = res_part(64, 64)
        self.res_part3_1 = res_part(64, 64)
        self.conv7_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu7_1 = nn.LeakyReLU(inplace=True)
        self.conv8_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.conv9_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu9_1 = nn.LeakyReLU(inplace=True)
        self.conv10_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        self.conv1_2 = nn.Conv2d(15, 32, kernel_size=5, stride=1, padding=2)
        self.relu1_2 = nn.LeakyReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.LeakyReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.relu3_2 = nn.LeakyReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.relu4_2 = nn.LeakyReLU(inplace=True)

        self.conv5_2 = nn.ConvTranspose2d(64*4, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5_2 = nn.LeakyReLU(inplace=True)
        self.conv51_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu51_2 = nn.LeakyReLU(inplace=True)
        self.conv52_2 = nn.Conv2d(32, 16, kernel_size=1, stride=1)
        self.relu52_2 = nn.LeakyReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)
        self.sigmoid6_2 = nn.Sigmoid()

        self.res_part1_2 = res_part(64, 64)
        self.res_part2_2 = res_part(64, 64)
        self.res_part3_2 = res_part(64, 64)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu7_2 = nn.LeakyReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu9_2 = nn.LeakyReLU(inplace=True)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

    def forward(self, x, mask, batch_size, meas_re, aug):
        maskt = mask.expand([batch_size, mask.shape[0], mask.shape[1], mask.shape[2]])
        maskt = maskt.mul(meas_re)
        # xt = torch.cat([meas_re, aug, maskt], dim=1)
        data1 = torch.cat([meas_re, aug[:,[0,1,3,4],:,:], maskt[:,:10,:,:]], dim=1)
        data2 = torch.cat([meas_re, aug[:,[0,1,3,4],:,:], maskt[:,10:,:,:]], dim=1)

        out_1 = self.conv1_1(data1)
        out_1 = self.relu1_1(out_1)
        out_1 = self.conv2_1(out_1)
        out_1 = self.relu2_1(out_1)
        out_1 = self.conv3_1(out_1)
        out_1 = self.relu3_1(out_1)
        out_1 = self.conv4_1(out_1)
        out_1_1 = self.relu4_1(out_1)

        out_1_2 = self.res_part1_1(out_1_1)
        out_1 = self.conv7_1(out_1_2)
        out_1 = self.relu7_1(out_1)
        out_1 = self.conv8_1(out_1)
        out_1_3 = self.res_part2_1(out_1)
        out_1 = self.conv9_1(out_1_3)
        out_1 = self.relu9_1(out_1)
        out_1 = self.conv10_1(out_1)
        out_1_4 = self.res_part3_1(out_1)

        out_1 = torch.cat([out_1_1,out_1_2,out_1_3,out_1_4],1)
        out_1 = self.conv5_1(out_1)
        out_1 = self.relu5_1(out_1)
        out_1 = self.conv51_1(out_1)
        out_1 = self.relu51_1(out_1)
        out_1 = self.conv52_1(out_1)
        out_1 = self.relu52_1(out_1)
        out_1 = self.conv6_1(out_1)

        out_2 = self.conv1_2(data2)
        out_2 = self.relu1_2(out_2)
        out_2 = self.conv2_2(out_2)
        out_2 = self.relu2_2(out_2)
        out_2 = self.conv3_2(out_2)
        out_2 = self.relu3_2(out_2)
        out_2 = self.conv4_2(out_2)
        out_2_1 = self.relu4_2(out_2)

        out_2_2 = self.res_part1_2(out_2_1)
        out_2 = self.conv7_2(out_2_2)
        out_2 = self.relu7_2(out_2)
        out_2 = self.conv8_2(out_2)
        out_2_3 = self.res_part2_2(out_2)
        out_2 = self.conv9_2(out_2_3)
        out_2 = self.relu9_2(out_2)
        out_2 = self.conv10_2(out_2)
        out_2_4 = self.res_part3_2(out_2)

        out_2 = torch.cat([out_2_1,out_2_2,out_2_3,out_2_4],1)
        out_2 = self.conv5_2(out_2)
        out_2 = self.relu5_2(out_2)
        out_2 = self.conv51_2(out_2)
        out_2 = self.relu51_2(out_2)
        out_2 = self.conv52_2(out_2)
        out_2 = self.relu52_2(out_2)
        out_2 = self.conv6_2(out_2)

        return out_1 , out_2


class cnn1(nn.Module):

    def __init__(self):
        super(cnn1, self).__init__()
        self.conv1 = nn.Conv2d(9, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.conv51 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu51 = nn.LeakyReLU(inplace=True)
        self.conv52 = nn.Conv2d(32, 16, kernel_size=1, stride=1)
        self.relu52 = nn.LeakyReLU(inplace=True)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.res_part1 = res_part(128, 128)
        self.res_part2 = res_part(128, 128)
        self.res_part3 = res_part(128, 128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.LeakyReLU(inplace=True)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=1, stride=1)

    def forward(self, x, mask, batch_size, meas_re):
        maskt = mask.expand([batch_size, 8, 256, 256])
        maskt = maskt.mul(meas_re)
        xt = torch.cat([meas_re, maskt], dim=1)
        data = xt
        out = self.conv1(data)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.res_part1(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.res_part2(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.conv10(out)
        out = self.res_part3(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)
        out = self.relu52(out)
        out = self.conv6(out)

        return out






class forward_rnn(nn.Module):

    def __init__(self):
        super(forward_rnn, self).__init__()
        h_size = 10
        x_h_size = 20
        flow_h_size = 4

        self.extract_feature1 = down_feature(1, x_h_size)
        self.extract_feature2 = flownet_feature(2, flow_h_size)
        self.extract_feature3 = aug_feature(6, x_h_size)
        self.up_feature1 = up_feature(h_size, 1)
        self.h_h = nn.Sequential(
            nn.Conv2d(h_size+x_h_size+x_h_size+flow_h_size, 20, 3, padding=1),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, h_size, 3, padding=1),
        )
        self.res_part1 = res_part(h_size+x_h_size+x_h_size+flow_h_size, h_size+x_h_size+x_h_size+flow_h_size)
        self.res_part2 = res_part(h_size+x_h_size+x_h_size+flow_h_size, h_size+x_h_size+x_h_size+flow_h_size)

    def forward(self, xt, aug, mask, batch_size, mode, meas_re, flow):
        h = torch.zeros_like(xt).cuda()

        out = torch.zeros_like(xt)
        out[:, 0, :, :] = torch.squeeze(xt[:,0,:,:])

        for i in range(xt.shape[1]-1):
            x_h = self.extract_feature1(torch.unsqueeze(xt[:,i,:,:],1))
            flow_h = self.extract_feature2(flow[:,i*2:(i+1)*2,:,:])
            aug_h = self.extract_feature3(torch.cat([aug,meas_re],1))

            h1 = torch.cat([h, x_h, aug_h, flow_h], dim=1)
            h2 = self.res_part1(h1)
            h3 = self.res_part2(h2)
            h = self.h_h(h3)
            x_t_1 = self.up_feature1(h)
            out[:,i+1,:,:] = torch.squeeze(x_t_1)

        return out



class backrnn(nn.Module):

    def __init__(self):
        super(backrnn, self).__init__()
        self.extract_feature1 = down_feature(1, 20)
        self.up_feature1 = up_feature(50, 1)
        self.conv_x = nn.Sequential(
            nn.Conv2d(2, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 80, 3, stride=2, padding=1),
            nn.Conv2d(80, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, 40, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(40, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.h_h = nn.Sequential(
            nn.Conv2d(50, 30, 3, padding=1),
            nn.Conv2d(30, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
        )
        self.res_part1 = res_part(50, 50)
        self.res_part2 = res_part(50, 50)

    def forward(self, xt8, meas, mask, batch_size, h, mode, meas_re):
        ht = h

        xt = xt8
        xt = torch.unsqueeze(xt, 1)

        out = torch.zeros(batch_size, 8, 256, 256).cuda()
        out[:, 7, :, :] = xt[:, 0, :, :]
        for i in range(7):
            d1 = torch.zeros(batch_size, 256, 256).cuda()
            d2 = torch.zeros(batch_size, 256, 256).cuda()
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask[7 - ii, :, :], out[:, 7 - ii, :, :].clone())
            for ii in range(i + 2, 8):
                d2 = d2 + torch.mul(mask[7 - ii, :, :], torch.squeeze(meas_re))
            x1 = self.conv_x(torch.cat([meas_re, torch.unsqueeze(meas - d1 - d2, 1)], dim=1))

            x2 = self.extract_feature1(xt)
            h = torch.cat([ht, x1, x2], dim=1)

            h = self.res_part1(h)
            h = self.res_part2(h)
            ht = self.h_h(h)
            xt = self.up_feature1(h)

            out[:, 6 - i, :, :] = xt[:, 0, :, :]

        return out
