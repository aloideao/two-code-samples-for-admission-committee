
import torch
from torch import nn
import torch.nn.functional as F
from basic import * 

class PANet(nn.Module):
  def __init__(self):
        super().__init__()
        #top-down
        self.c5_spp=SPP_Block()
        self.c5_conv=conv_block(512,256,1)
        self.upsample=nn.Upsample(scale_factor=2)

        self.c4_conv_1=conv_block(512,256,1)
        self.c4_blocks=five_conv_blocks(512)
        self.c4_conv_2=conv_block(256,128,1)

        self.c3_conv=conv_block(256,128,1)
        self.c3_blocks=five_conv_blocks(256)
        #down-top
        self.c3_conv_downsample=conv_block(128,256,3,stride=2)
        self.c4_convs_td=five_conv_blocks(512)
        self.c4_conv_downsample=conv_block(256,512,3,stride=2)
        self.c5_convs_td=five_conv_blocks(1024)


        self.c5_head=conv_block(512,1024,3)
        self.c4_head=conv_block(256,512,3)
        self.c3_head=conv_block(128,256,3)

  def forward(self,x):
    #top-down
    c5,c4,c3=x
    c5=self.c5_spp(c5) #b 512 13 13
    c5_route=c5 #b 512 13 13
    c5=self.c5_conv(c5) #b 256 13 13

    c5_upsampled=self.upsample(c5) #b 256 26 26

    c4=self.c4_conv_1(c4) #b 256 26 26
    c4=torch.cat([c4,c5_upsampled],1)   #b  512 26 26
    c4=self.c4_blocks(c4) #b  256 26 26
    c4_route=c4  #b  256 26 26
    c4=self.c4_conv_2(c4) #b  128 26 26
    c4_upsampled=self.upsample(c4) #b 128 52 52

    c3=self.c3_conv(c3) #b 128 52 52
    c3=torch.cat([c3,c4_upsampled],1) #b 256 52 52
    c3=self.c3_blocks(c3) #b 128 52 52
    c3_route=c3

    #down-top

    c3=self.c3_conv_downsample(c3) #b 256 26 26
    c4=torch.cat([c4_route,c3],1) #b 512 26 26
    c4=self.c4_convs_td(c4) #b 256 26 26
    c4_route=c4  #b 256 26 26

    c4=self.c4_conv_downsample(c4) #b 512 13 13

    c5=torch.cat([c5_route,c4],1) #b 1024 13 13
    c5_route=self.c5_convs_td(c5) #b 512 13 13

    c5=self.c5_head(c5_route) #b 1024 13 13
    c4=self.c4_head(c4_route) #b 512 26 26
    c3=self.c3_head(c3_route) #b 256 52 52

    return c5,c4,c3

