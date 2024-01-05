
import torch
from torch import nn
import torch.nn.functional as F
from backbone import CSPDarknet53
from head import Yolohead
from PANet import PANet

class Yolov4(nn.Module):
    def __init__(self,strides=[8,16,32]):

      anchors=[[(12, 16), (19, 36), (40, 28)],
              [(36, 75), (76, 55), (72, 146)],
              [(142, 110), (192, 243), (459, 401)]]
      super().__init__()
      self.anchors=torch.tensor(anchors)
      self.strides=strides

      self.backbone=CSPDarknet53()
      self.neck=PANet()
      self.head_p3=Yolohead(256,self.anchors[0],self.strides[0])
      self.head_p4=Yolohead(512,self.anchors[1],self.strides[1])
      self.head_p5=Yolohead(1024,self.anchors[2],self.strides[2])

    def forward(self,x):
        x=self.backbone(x) #x=[c5,c4,c3]
        c5,c4,c3=self.neck(x)
        c3=self.head_p3(c3)
        c4=self.head_p4(c4)
        c5=self.head_p5(c5)
        return c3,c4,c5


if __name__=='__main__':
   model=Yolov4()
   print(model(torch.randn(1,3,416,416)))
   