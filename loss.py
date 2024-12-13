import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=20):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target):
        distances = torch.norm(output1 - output2, p=2, dim=1)
        losses = target * torch.square(distances) + (1 - target) * torch.square(torch.clamp(self.margin - distances, min=0))    
        return losses.sum(), distances


