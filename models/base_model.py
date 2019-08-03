import torch
from torch import nn
from torch.nn import functional as F






class _BaseModel(nn.Module):
    '''

    Attributes:
        encoder: CNN 
        classify: MLP + Sigmoid

    Method:
        get_feature_map: Take FloatTensor Images (shape: b, 1, w, h), return feature map before flatten
        forward: Full flow: Images -> Encoder -> Flatten -> MLP -> Sigmoid -> Labels


    '''

    def __init__(self, nClass, pretrain=True):
        super(_BaseModel, self).__init__()
        self.encoder = None
        self.classify = None

    def get_feature_map(self, imgs):
        return self.encoder(imgs)
    
    def forward(self, imgs):
        fm = self.encoder(imgs)
        b = fm.shape[0]
        fm = fm.view(b, -1)
        return self.classify(fm)
