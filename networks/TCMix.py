from unetr import UNETR
from monai.networks.nets import SegResNet
import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks import ChannelSELayer
from monai.networks.layers.factories import Act, Norm
from typing import Optional, Sequence, Tuple, Union
from monai.networks.blocks.dynunet_block import get_conv_layer,get_padding,get_output_padding


class TCMix(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        use_multi_mix=False
    ) -> None:
        
        
        super().__init__()
        self.align_channel = _make_align_channel(3,8,16)
        self.Transformer=UNETR(
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
            
        self.CNN=SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            upsample_mode="deconv",
            use_conv_final=False,
            using_feature=True)
    def _make_align_channel(self,spatial_dim,in_channel,out_channel):
        return get_conv_layer(spatial_dims=spatial_dim,in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1)
    
    def _make_mix_upsample(self,feat1,feat2,spatial_dim,in_channel,out_channel):
        return nn.Sequential(
        torch.cat((feat1,feat2),1)
        ChannelSELayer(spatial_dim,out_channel, r=2, acti_type_1=('relu', {'inplace': True}), acti_type_2='sigmoid', add_residual=False)
        get_norm_layer(name=("group", {"num_groups": 32}))
        )# concat-SE-BN
    
        
        self.mixup_layer=get_conv_layer(spatial_dims=3,in_channels=32,out_channels=16,kernel_size=3,stride=1) # add a BN here
        self.out = UnetOutBlock(spatial_dims=3, in_channels=16, out_channels=2)  # type: ignore

    def forward(self, x_in):
        CNNFeature=self.CNN(x_in)
        TransformerFeature=self.Transformer(x_in)
        CNNFeature=self.align_channel(CNNFeature[2])
        Mixed=torch.cat((CNNFeature,TransformerFeature[2]),1)
        Mixed=self.mixup_layer(Mixed)
        Mixed=self.out(Mixed)
        return Mixed

if __name__ =='__main__':
    device = torch.device('cuda')
    model=TCMix().to(device)
    dummy=torch.randn(1, 1, 96, 96,96).float().to(device)
    x=model(dummy)
    print(x.shape)
        
        
        