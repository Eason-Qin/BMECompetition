import os
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete,Activations,Compose
from networks.unetr import UNETR
from monai.networks.nets import UNet
from monai.networks.nets import SegResNet
from utils.data_utils import get_loader
from trainer import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
import argparse

if __name__ =='__main__':
    torch.cuda.empty_cache() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict=torch.load("/home/robotlab/research-contributions/UNETR/BTCV/runs/segres_S/model_final.pth")
    model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            upsample_mode="deconv",
            use_conv_final=False)
    model_state_dict = model.state_dict()
    state_dict = {k:v for k,v in model_dict.items() if k in model_state_dict.keys()}
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    dummy=torch.randn(1, 1, 96,96,96).float().to(device)
    x=model(dummy)
    dummy1=torch.randn(1, 16, 96,96,96).float().to(device)
    dummy2=torch.randn(1, 16, 96,96,96).float().to(device)
    test=torch.cat((dummy1,dummy2),1)
    print(test.shape)

