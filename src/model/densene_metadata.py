import torch.nn as nn
import torchvision.models as models

class DenseNetWithMetadata(nn.Module):
    def __init__(self, metadata_input_size=3):  
        super(DenseNetWithMetadata, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Identity()  

        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, image, metadata):
        image_features = self.densenet(image)
        metadata_features = self.metadata_fc(metadata)
        combined = torch.cat([image_features, metadata_features], dim=1)
        out = self.classifier(combined)
        return out
