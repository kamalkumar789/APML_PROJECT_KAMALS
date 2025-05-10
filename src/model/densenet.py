import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        in_features = self.densenet.classifier.in_features

        # Add dropout before the final classifier
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)  # For binary classification
        )

    def forward(self, x):
        return self.densenet(x)
