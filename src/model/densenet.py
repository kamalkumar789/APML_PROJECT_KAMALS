import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(weights=None)
        in_features = self.densenet.classifier.in_features

        # Add dropout before the final classifier
        self.densenet.classifier = nn.Sequential(
            nn.dropout(0.50),
            nn.Linear(in_features, 1)  # For binary classification
        )

    def forward(self, x):
        return self.densenet(x)
