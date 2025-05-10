import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Make sure inputs are float and targets are long for indexing
        inputs = inputs.view(-1)
        targets = targets.view(-1).long()

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0

        # If alpha provided, apply per-class weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha).to(inputs.device)
            elif isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
            else:
                raise TypeError("alpha must be a list, tuple or torch.Tensor")

            at = alpha[targets]  # now safe because targets is LongTensor
            BCE_loss = at * BCE_loss

        focal_loss = ((1 - pt) ** self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
