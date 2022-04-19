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
from utils.data_utils import get_loader
from trainer import dice
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from utils.data_utils import get_loader
from utils.valid_utils import dice
from utils.visualization import print_cut_samples,print_heatmap
from utils.slide_saliency_inference import sliding_saliency_inference

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='./dataset/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--save_img', default='test', type=str, help='image cut save file')
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
    dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:23456',
                                world_size=1,
                                rank=0)
    torch.cuda.empty_cache() 
    args = parser.parse_args()
    # enable test mode
    args.test_mode = True
    # load data
    val_loader = get_loader(args)
    # load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    sf=SaliencyInferer("CAM","out.conv.conv")
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model=torch.load(pretrained_pth)
    '''model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate)'''
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

    '''model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    model.load_state_dict(model_dict["state_dict"])'''
    model.eval()
    model.to(device)
    start_time = time.time()
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            # sliding windows inference
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            
            path=img_name+os.path.split(pretrained_pth,".")[0]
            print_cut_samples(val_inputs,val_labels,val_outputs,path)
            salient=sliding_saliency_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   sf,
                                                   model,
                                                   overlap=args.infer_overlap)
            path=img_name+os.path.split(pretrained_pth,".")[0]+"heatmap"
            print_heatmap(val_inputs,val_labels,val_outputs,salient,path)
            # postprocess softmax->argmax
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            # calculate desired metric
            organ_Dice = dice(val_outputs[0] == 1, val_labels[0] == 1)
            dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            torch.cuda.empty_cache()
        # ave. metric.
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':
    main()