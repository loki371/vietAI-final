from .base_model import _BaseModel
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils

from torch import nn
from torch.nn import functional as F

class EffNet(_BaseModel):
    def __init__(self, nClass, mode='efficientnet-b2'):
        super(EffNet, self).__init__(nClass, pretrain=True)
        model = EfficientNet.from_pretrained(mode) 
        blocks_args, global_params = utils.get_model_params(mode, None)
        model._conv_stem = utils.Conv2dStaticSamePadding(1, 32, kernel_size=(3, 3), stride=(2, 2), image_size=global_params.image_size)
        
        model._fc = nn.Linear(model._fc.in_features, nClass)

        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.classify = nn.Sequential(
            model._fc,
            nn.Sigmoid()
        )
        self.model = model

    def get_feature_map(self, imgs):
        return self.model.extract_features(imgs)

    def forward(self, imgs):
        return F.sigmoid(self.model(imgs))
