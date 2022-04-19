from unetr import UNETR
from segres import SegResNet
import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks import ChannelSELayer
from monai.networks.layers.factories import Act, Norm
from typing import Optional, Sequence, Tuple, Union
from monai.networks.blocks.dynunet_block import get_padding,get_output_padding,get_norm_layer
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from PytorchMemoryUtils.gpu_mem_track import MemTracker
from PytorchMemoryUtils.modelsize_estimate import modelsize

class TCMix(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self,_use_multi_mix=False) -> None:
        
        
        super().__init__()
        self.use_multi_mix=_use_multi_mix
        self.align_channel_3 = self._make_align_channel(3,32,64)
        self.align_channel_2 = self._make_align_channel(3,16,32)
        self.align_channel_1 = self._make_align_channel(3,8,16)
        self.SE_3 = ChannelSELayer(3,64*2, r=2, acti_type_1=('relu', {'inplace': True}), acti_type_2='sigmoid', add_residual=False)
        self.SE_2 = ChannelSELayer(3,32*3, r=2, acti_type_1=('relu', {'inplace': True}), acti_type_2='sigmoid', add_residual=False)
        self.SE_1 = ChannelSELayer(3,16*3, r=2, acti_type_1=('relu', {'inplace': True}), acti_type_2='sigmoid', add_residual=False)
        self.GN_3 = get_norm_layer(name=("group", {"num_groups": 16}),spatial_dims=3, channels=32)
        self.GN_1 = get_norm_layer(name=("group", {"num_groups": 16}),spatial_dims=3, channels=16)
        self.mixup_layer=get_conv_layer(spatial_dims=3,in_channels=32,out_channels=16,kernel_size=3,stride=1) # add a BN here
        self.out = UnetOutBlock(spatial_dims=3, in_channels=16, out_channels=2)  # type: ignore
        self.UNETR=UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96,96,96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0,
            use_feature=True)
            
        self.SegResNet=SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            upsample_mode="deconv",
            use_conv_final=False,
            using_features=True)
            
        self.upsample_3=nn.Sequential(
                        get_conv_layer(3, 2*64, 32, kernel_size=1),
                        get_upsample_layer(3, 32, upsample_mode='deconv'),)
        self.upsample_2=nn.Sequential(
                        get_conv_layer(3, 3*32, 16, kernel_size=1),
                        get_upsample_layer(3,16, upsample_mode='deconv'),)
        self.upsample_1=nn.Sequential(
                        get_conv_layer(3, 3*16, 16, kernel_size=1),
                        )# reduce channel is fine
            
    def _make_align_channel(self,spatial_dim,in_channel,out_channel):
        return get_conv_layer(spatial_dims=spatial_dim,in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1)
    
    def mix_upsample_bottom(self,feat1,feat2,SEBlock,in_channel,out_channel):
        feat_cat = torch.cat((feat1,feat2),1)
        feat_cat = SEBlock(feat_cat)
        feat_cat = self.upsample_3(feat_cat)
        feat_cat = self.GN_3(feat_cat)
        return feat_cat
    
    def mix_upsample(self,feat1,feat2,feat3,SEBlock,upsampleBlock,in_channel,out_channel):
        feat_cat = torch.cat((feat1,feat2,feat3),1)
        feat_cat = SEBlock(feat_cat)
        feat_cat = upsampleBlock(feat_cat)
        feat_cat = self.GN_1(feat_cat)
        return feat_cat
        # concat-SE-deconv-BN
    
        
    

    def forward(self, x_in):
        if not self.use_multi_mix:
            CNNFeature=self.SegResNet(x_in)
            TransformerFeature=self.UNETR(x_in)
            CNNFeature=self.align_channel(CNNFeature[2])
            Mixed=torch.cat((CNNFeature,TransformerFeature[2]),1)
            Mixed=self.mixup_layer(Mixed)
            Mixed=self.out(Mixed)
            return Mixed
        if self.use_multi_mix:
            CNNFeature=self.SegResNet(x_in)
            TransformerFeature=self.UNETR(x_in)
            mix1=self.align_channel_1(CNNFeature[2])#8-16
            mix2=self.align_channel_2(CNNFeature[1])#16-32
            mix3=self.align_channel_3(CNNFeature[0])#32-64
            up_mix_3=self.mix_upsample_bottom(mix3,TransformerFeature[0],self.SE_3,128,32)
            up_mix_2 = self.mix_upsample(mix2,TransformerFeature[1],up_mix_3,self.SE_2,self.upsample_2,32,16)
            up_mix_1 = self.mix_upsample(mix1,TransformerFeature[2],up_mix_2,self.SE_1,self.upsample_1,16,16)
            out = self.out(up_mix_1)
            return out
            
            
            
            
            

if __name__ =='__main__':
    device = torch.device('cuda:1')
    gpu_tracker = MemTracker()
    model=TCMix(_use_multi_mix=True).to(device)
    model.train()
    gpu_tracker.track()
    dummy=torch.randn(1, 1, 96, 96, 96).float().to(device)
    x=model(dummy)
    gpu_tracker.track()
    #modelsize(model,dummy)
    #print(x.shape)
    for name, param in model.named_parameters():
        print(name)
        
        
        