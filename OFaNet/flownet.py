import torch
import torch.nn as nn
import numpy as np
import argparse
from models_flownet import FlowNet2
from utils.frame_utils import read_gen
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)
args = parser.parse_args()

def expand_channel(x1,x2):
    img1 = np.zeros([256, 256, 3])
    img2 = np.zeros([256, 256, 3])

    img1_single = x1
    img2_single = x2

    for i in range(3):
        img1[:, :, i] = img1_single
        img2[:, :, i] = img2_single

  #  img1 = cv2.resize(img1, dsize=(648, 648))
  #  img2 = cv2.resize(img2, dsize=(648, 648))
    return img1, img2

class flownet(nn.Module):
    def __init__(self):
        super(flownet,self).__init__()
        self.model = FlowNet2(args).cuda()
        # dict = torch.load("FlowNet2_checkpoint.pth.tar")
        # self.model.load_state_dict(dict["state_dict"])

    def forward(self,x):
        batch_size = x.size()[0]
        channel = x.size()[1]
        single_flow = np.zeros([(channel - 1) * 2,x.size()[2],x.size()[2]])
        batch_flow = np.zeros([batch_size,(channel - 1) * 2,x.size()[2],x.size()[2]])

        for i in range(batch_size):
            for j,k in enumerate(list(range(0,(channel - 1) * 2,2))):
                x_t = x[i][j].data.cpu().numpy()
                x_t1 = x[i][j + 1].data.cpu().numpy()
                x_t,x_t1 = expand_channel(x_t,x_t1)
                input = [x_t, x_t1]
                input = np.array(input).transpose(3, 0, 1, 2)
                im = torch.from_numpy(input.astype(np.float32)).unsqueeze(0).cuda()
                 # process the image pair to obtian the flow
                output = self.model(im).squeeze()
                output = output.data.cpu().numpy().transpose(1, 2, 0)
              #  output = cv2.resize(output,dsize = (650,650))
                output = np.transpose(output,(2,0,1))
                # output = output.data.cpu().numpy()
                single_flow[k] = output[0]
                single_flow[k + 1] = output[1]
            batch_flow[i] = single_flow

        batch_flow = torch.from_numpy(batch_flow).float().cuda()
        return batch_flow

if __name__ == '__main__':
    flownet = flownet()
