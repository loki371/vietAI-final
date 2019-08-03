from torchvision.models import resnet50
from .base_model import _BaseModel
from torch import nn

class ResNet50(_BaseModel):
    def __init__(self, nClass, pretrain=True):
        super(ResNet50, self).__init__(nClass, pretrain)
        resnet = resnet50(pretrain)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        
        self.classify = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, nClass),
            nn.Sigmoid()
        )

