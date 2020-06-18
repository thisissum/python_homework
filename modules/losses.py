from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):
    """Focal loss implemented for F1 optim task
    params:
        alpha: float, weight of each class, default 0.25
        gamma: float, param to tune hard and easy sample, default 2
        num_classes: int, num of class to classify
        size_average: bool, True for mean, False for sum, default True
    """
    def __init__(self, num_classes, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = torch.FloatTensor([alpha] + [1-alpha]*(num_classes-1))
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, y_pred, y_true):
        # shape(y_pred) = batch_size, labels, ...
        # shape(y_true) = batch_size, output_dim,labels_dim...
        y_true = y_true.unsqueeze(-1)
        y_pred_proba = F.softmax(y_pred, dim=1)
        y_pred_log_proba = torch.log(y_pred_proba)
        y_pred_proba = torch.gather(y_pred_proba, dim=1, index=y_true)
        y_pred_log_proba = torch.gather(y_pred_log_proba, dim=1, index=y_true)
        self.alpha = self.alpha.to(y_pred.device).gather(0,y_true.view(-1))
        loss = -torch.mul(torch.pow(1-y_pred_proba, self.gamma), y_pred_log_proba)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            return loss.mean()
        else: 
            return loss.sum()
