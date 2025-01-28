import torch

def dice_score(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 5).float()
    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + 1e-6) / (y_pred.sum() + y_true.sum() + 1e-6) 
