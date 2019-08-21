from torchvision.models import resnet50
from .base_model import _BaseModel
from torch import nn

class ResNet50(_BaseModel):
    def __init__(self, nClass, pretrain=True):
        super(ResNet50, self).__init__(nClass, pretrain)
        resnet = resnet50(pretrain)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.encoder = nn.Sequential(resnet.features, resnet.avgpool)
        
        self.classify = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, nClass),
            nn.Sigmoid()
        )

