import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, 1)  

    def forward(self, x):
        return self.densenet(x) 