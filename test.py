# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from networks.unetr import UNETR
from utils.data_utils import get_loader
from trainer import dice
import argparse
import os
import matplotlib.pyplot as plt
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather
import torch.utils.data.distributed
from monai.data import decollate_batch
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
from utils.data_utils import get_loader
from trainer import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='./dataset/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=5, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')

def main():
    torch.cuda.empty_cache() 
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model=torch.load(pretrained_pth)
    inf_size = [args.roi_x, args.roi_y, args.roi_x]
    post_label = AsDiscrete(to_onehot=True,
                            n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=True,
                           n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)
    model_inferer = partial(sliding_window_inference,
                            roi_size=inf_size,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=args.infer_overlap)

    model.eval()
    model.to(device)
    start_time = time.time()
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title("image")
            cut = 66
            plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, cut], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title("label")
            plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, cut])
            plt.subplot(1, 3, 3)
            plt.title("output")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, cut])
            plt.savefig("test1",dpi=300)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            organ_Dice = dice(val_outputs[0] == 1, val_labels[0] == 1)
            dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            torch.cuda.empty_cache()
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        '''for idx, batch_data in enumerate(val_loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']
            img_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print(img_name)
            print(data.shape)
            print(target.shape)
            data, target = data.cuda(0), target.cuda(0)
            with autocast(enabled=False):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title("image")
            cut = 66
            plt.imshow(data.cpu().numpy()[0, 0, :, :, cut], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title("label")
            plt.imshow(target.cpu().numpy()[0, 0, :, :, cut])
            plt.subplot(1, 3, 3)
            plt.title("output")
            plt.imshow(torch.argmax(logits, dim=1).detach().cpu()[0, :, :, cut])
            plt.savefig("test1",dpi=300)
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = dice_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(0)
            if args.distributed:
                acc_list = distributed_all_gather([acc],
                                                  out_numpy=True,
                                                  is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
            print('acc', avg_acc,'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()
            torch.cuda.empty_cache()'''

if __name__ == '__main__':
    main()