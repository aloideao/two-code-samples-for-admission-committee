
import torch
from torch import nn
import torch.nn.functional as F

class conv_block(nn.Module):
  def __init__(self,in_channels,out_channels,kernel,stride=1,padding_mode='same',activation='LeakyReLU',**kwargs):
    super().__init__()
    self.in_channels=in_channels
    self.conv_layer=nn.Sequential(
        nn.Conv2d(in_channels=self.in_channels,out_channels=out_channels,kernel_size=kernel,stride=stride,padding=kernel//2 if padding_mode == 'same' else 0,bias=False,**kwargs ),
        nn.BatchNorm2d(out_channels))
    self.activation=nn.LeakyReLU(inplace=True) if activation=='LeakyReLU' else nn.Mish(inplace=True)

  def forward(self,x):
    return self.activation(self.conv_layer(x))

class resblock(nn.Module):
  def __init__(self,in_channels,num_blocks):
    super().__init__()
    self.res_layer=nn.Sequential(conv_block(in_channels,in_channels//2,1,activation='Mish'),
                 conv_block(in_channels//2,in_channels,3,activation='Mish'))
  def forward(self,x):
    return x+self.res_layer(x)

class csp_block(nn.Module):
    def __init__(self,feat,num_of_blocks):
      super().__init__()
      self.stage_1=conv_block(feat,feat//2,1,activation='Mish')
      self.stage_2=nn.Sequential(conv_block(feat,feat//2,1,activation='Mish'),
                                resblock(feat//2,num_of_blocks),
                                conv_block(feat//2,feat//2,1,activation='Mish'))
      self.conv=conv_block(feat,feat,1,activation='Mish')
    def forward(self,x):
      one=self.stage_1(x)
      two=self.stage_2(x)
      x=torch.cat([one,two],1)
      x=self.conv(x)
      return x
class five_conv_blocks(nn.Module):
    def __init__(self,feat):
      super().__init__()
      self.convs=nn.Sequential(conv_block(feat,feat//2,1),
                               conv_block(feat//2,feat,3),
                               conv_block(feat,feat//2,1),
                               conv_block(feat//2,feat,3),
                               conv_block(feat,feat//2,1))
    def forward(self,x):
        return self.convs(x)



class SPP(nn.Module):
    def __init__(self):
      super().__init__()
    def forward(self,x):
        x1=F.max_pool2d(x,kernel_size=5,stride=1,padding=2)
        x2=F.max_pool2d(x,kernel_size=9,stride=1,padding=4)
        x3=F.max_pool2d(x,kernel_size=13,stride=1,padding=6)
        return torch.cat([x,x1,x2,x3],1)
class SPP_Block(nn.Module):
      def __init__(self,feat=1024):
       super().__init__()
       self.spp_block=nn.Sequential(
                      conv_block(feat,feat//2,1),
                      conv_block(feat//2,feat,3),
                      conv_block(feat,feat//2,1),
                      SPP(),
                      conv_block(feat//2*4,feat//2,1),
                      conv_block(feat//2,feat,3),
                      conv_block(feat,feat//2,1),
                      )
      def forward(self,x):
        return  self.spp_block(x)

