# OFaNet-for-Dual-view-SCI
This repository contains the code for the paper "Dual-view Snapshot Compressive Imaging via Optical Flow Aided Recurrent Neural Network" by Ruiying Lu, Bo Chen, Guanliang Liu, Ziheng Cheng, Mu Qiao and Xin Yuan.

Please download the OFanet.zip and uncompressed it.

## Requirements Enverionment
download flownet.yml and install it on your computer 

## Data
The training data for OFaNet is generated from [DAVIS2017](https://davischallenge.org/davis2017/code.html) with random crop and final obtain 32,000 data pairs. If you want to use the same training data as ours, please run ```generate_simu_data_dualview_with_background.m``` in MATLAB.

The simulation test data including six simulation data and corresponding masks are placed in the ```OFaNet/test_data``` folder.

## Pretrained model
You can download the models we trained for simulation dataset from https://pan.baidu.com/s/1Chh26em3x9FMdEDJMT-SBg 
code：mkjc.

## Train
Run
```
python train.py
```

## Test
Run
```
python test.py
```
Using it to evaluate the preformance on simulation data and we will release the pre-trained model.

## Contact
[RuiyingLu, Xidian University](mail to: ruiyinglu_xidian@163.com) 
