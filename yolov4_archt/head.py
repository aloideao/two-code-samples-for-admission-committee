
import torch
from torch import nn
import torch.nn.functional as F

class Yolohead(nn.Module):
    def __init__(self,feat,anchors,stride,classes=20):
      super().__init__()
      self.num_anchors=len(anchors)
      self.classes_num=classes
      self.head=nn.Conv2d(feat,(5+self.classes_num)*self.num_anchors,1)
      self.stride=stride
      self.anchors=anchors
    @staticmethod
    def grid(size):
      grid_xy=torch.stack(torch.meshgrid([torch.arange(size),torch.arange(size)],indexing='ij'),-1)
      return grid_xy
    def forward(self,x):
      b,_,h,w=x.shape
      prediction=self.head(x).contiguous().view(b,self.num_anchors,self.classes_num+5,h,w).permute(0,3,4,1,2)
      pred_xy=(torch.sigmoid(prediction[...,0:2])+self.grid(h).unsqueeze(-2))*self.stride
      pred_Wh=torch.exp(prediction[...,2:4])*self.anchors
      pred_conf=prediction[...,4:5].sigmoid()
      pred_classes=prediction[...,5:].sigmoid()
      pred_bbox=torch.cat([pred_xy,pred_Wh,pred_conf,pred_classes],-1)

      return pred_bbox.view(b,-1,self.classes_num+5) if not self.training else pred_bbox


