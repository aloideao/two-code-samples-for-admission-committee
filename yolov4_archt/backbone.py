
import torch
from torch import nn
import torch.nn.functional as F
from basic import * 
class CSPDarknet53(nn.Module):
    def __init__(self,num_classes=5):

        super().__init__()
        self.layer_1=nn.Sequential(
                                        conv_block(3,32,3,activation='Mish'),
                                        conv_block(32,64,3,stride=2,activation='Mish'),
                                        csp_block(64,1))
        self.layer_2=nn.Sequential(
                                        conv_block(64,128,3,stride=2,activation='Mish'),
                                        csp_block(128,2))
        self.layer_3=nn.Sequential(
                                        conv_block(128,256,3,stride=2,activation='Mish'), #p3
                                        csp_block(256,8))
        self.layer_4=nn.Sequential(
                                        conv_block(256,512,3,stride=2,activation='Mish'), #p4
                                        csp_block(512,8))
        self.layer_5=nn.Sequential(
                                        conv_block(512,1024,3,stride=2,activation='Mish'), #p5
                                        csp_block(1024,4))
        self.classifier=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
                )
        self._init()

    def _init(self):
        for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)


    def forward(self,x):

            c1 = self.layer_1(x)
            c2 = self.layer_2(c1)
            c3 = self.layer_3(c2)
            c4 = self.layer_4(c3)
            c5 = self.layer_5(c4)
            if False:
              output=self.classifier(c5)
            else:
              output=[c5,c4,c3]
            return output

